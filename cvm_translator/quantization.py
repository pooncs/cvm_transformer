import torch
from torch.ao.quantization import QuantStub, DeQuantStub


class QuantizedCVMTransformer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, input_ids, core_indices=None, core_kv=None):
        x = self.quant(input_ids)
        x = self.model(x, core_indices, core_kv)
        return self.dequant(x)


def prepare_qat(model):
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(model, inplace=True)


def convert_int8(model):
    torch.ao.quantization.convert(model, inplace=True)


if __name__ == "__main__":
    from cvm_translator.cvm_transformer import CVMTransformer
    m = CVMTransformer(vocab_size=1000, d_model=128, n_layers=2)
    qm = QuantizedCVMTransformer(m)
    prepare_qat(qm)
    print("QAT prepared")