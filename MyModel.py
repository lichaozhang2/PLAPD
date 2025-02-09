import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import esm
import math
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('encoding', self._get_timing_signal(max_len, d_model))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

    def _get_timing_signal(self, length, channels):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * -(math.log(10000.0) / channels))
        pe = torch.zeros(length, channels)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_len=5000):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward)

    def forward(self, src, tgt):
        src = src + self.pos_encoder(src)
        # src = self.pos_encoder(src)
        output = self.transformer(src, tgt)
        return output


class ESM(nn.Module):
    def __init__(self):
        super(ESM, self).__init__()
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

    def forward(self, prot_seqs):
        data = [('seq{}'.format(i), seq) for i, seq in enumerate(prot_seqs)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        # Extract per-residue representations (on GPU)
        with torch.no_grad():
            results = self.esm_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33][:, 1:-1]

        # [batch, L, 1280]
        prot_embedding = token_representations

        return prot_embedding


class AIMP(torch.nn.Module):
    def __init__(self, pre_feas_dim, hidden, n_transformer, dropout):
        super(AIMP, self).__init__()

        self.esm = ESM()

        self.pre_embedding = nn.Sequential(
            nn.Conv1d(pre_feas_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.bn = nn.ModuleList([nn.BatchNorm1d(pre_feas_dim)])

        self.n_transformer = n_transformer

        self.transformer = TransformerModel(d_model=hidden, nhead=4, num_layers=self.n_transformer,
                                            dim_feedforward=2048)
        self.transformer_act = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.transformer_res = nn.Sequential(
            nn.Conv1d(hidden + hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.transformer_pool = nn.AdaptiveAvgPool2d((1, None))
        self.clf = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.pre_embedding, self.clf]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)
        for layer in [self.transformer_act, self.transformer_res]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)

    def forward(self, protein_sequence):

        bert_output = self.esm(protein_sequence)

        pre_feas = self.bn[0](bert_output.permute(0, 2, 1)).permute(0, 2, 1)

        pre_feas = self.pre_embedding(pre_feas.permute(0, 2, 1)).permute(0, 2, 1)

        transformer_out = self.transformer(pre_feas, pre_feas)
        transformer_out = self.transformer_act(transformer_out.permute(0, 2, 1)).permute(0, 2, 1)
        transformer_out = self.transformer_res(torch.cat([transformer_out, pre_feas], dim=-1).permute(0, 2, 1)).permute(
            0, 2, 1)
        transformer_out = self.transformer_pool(transformer_out).squeeze(1)

        out = self.clf(transformer_out)
        out = torch.nn.functional.softmax(out, -1)
        return out, bert_output, pre_feas, transformer_out

# if __name__ == '__main__':
#     protein_sequences = [
#         "MKTAYIAKQRQISFVKSHFSRQDILDLIQKQKFKQVDLRQQVKQHSLTVR",
#         "MPQNKVLSFGLKDEKDGEKLVFNGLRLVSEAPSGDQLQLLRKFLKHLQDRF"
#     ]
#
#     model = AIMP(pre_feas_dim=1280, hidden=1280, n_transformer=1, dropout=0.5)
#     model.cuda()
#     # 生成一个形状为[1, 30, 1024]的随机tensor
#     random_tensor = torch.randn(2, 30, 1024)
#     model(protein_sequences)
