import torch

model = " "
state_dict_path = " "
state_dict = torch.load(state_dict_path, map_location='cpu')
    
if 'model' in state_dict:
    checkpoint = state_dict['checkpoint']
else:
    checkpoint = state_dict

for key in list(state_dict.keys()): # adjust different names in pretrained checkpoint
    if 'bert' in key:
        encoder_key = key.replace('bert.', '')
        state_dict[encoder_key] = state_dict[key]
        del state_dict[key]

print("Start loading form the checkpoint......")
msg = model.load_state_dict(state_dict,strict=False)
assert len(msg.missing_keys) == 0
