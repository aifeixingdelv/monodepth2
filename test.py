import networks
from timm.models import create_model

model = create_model('mpvim_tiny', pretrained=True)
print(model)