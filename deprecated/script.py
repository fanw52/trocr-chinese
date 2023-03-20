import torch
# path = "pre-train.pt"
# data = torch.load(path,map_location='cpu')
# for k,v in data.items():
#     print(k,v.shape)


path = "../cust-data/trocr-small-handwritten.pt"
data = torch.load(path,map_location='cpu')

model = data["trocr"]


#
# new_model = {}
for k,v in model.items():
    print(k,v.shape)
#     if k == "decoder.output_projection.weight" or k == "decoder.embed_tokens.weight":
#         continue
#     new_model[k] = v
#
# torch.save(new_model, "./pre-train.pt")
# print()
#
# #
# print()
# path = "../cust-data/pytorch_model.bin"
# model1 = torch.load(path,map_location='cpu')
# for k,v in model1.items():
#     print(k,v.shape)
# for k,v in trocr.items():
#     print(k,v.shape)
#
#
# print(len(model1))
# print(len(trocr))
# for (k1,v1),(k2,v2) in zip(model1.items(),trocr.items()):
#     print(k1,v1.shape)
#     print(k2,v2.shape)
#     print()
# new_model = {}
# for k,v in trocr.items():
#     if k == "decoder.output_projection.weight" or k=="decoder.embed_tokens.weight":
#         continue
#     new_model[k] = v
#
# torch.save(new_model,"./hand.pt")