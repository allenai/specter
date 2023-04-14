import torch  
from allennlp.nn import util


cuda_device = -1
weights_file = "/tmp/tmpxizb5pwo/weights.th" #scibert
#weights_file = "/tmp/tmpxizb5pwo/weights.th" #finbert



def load_model():
  
  
  print("[DEBUG] weights file: ", weights_file)
  model_state = torch.load(weights_file, map_location=util.device_mapping(cuda_device))
  print("[DEBUG] model_state", model_state.keys())
  model.load_state_dict(model_state)

  if cuda_device >= 0:
          model.cuda(cuda_device)
  else:
      model.cpu()



if __name__ == "__main__":
    print("LOADING MODEL")
    model = load_model()
    print("MODEL LOADED")
