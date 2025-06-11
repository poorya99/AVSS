import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F

# ---------------------------------------------------
# 1. Device Configuration
# ---------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------------------
# 2. Load the CASIA-WebFace pretrained FaceNet model
# ---------------------------------------------------
casia_model = InceptionResnetV1(
    classify=False,
    pretrained='casia-webface'
).to(device)  # Move model to the selected device
casia_model.eval()  # Set to evaluation mode

# ---------------------------------------------------
# 3. Freeze the facenet-pytorch CASIA model
# ---------------------------------------------------
for param in casia_model.parameters():
    param.requires_grad = False

# ---------------------------------------------------
# 4. Define a deeper MLP to map face embeddings to 256-D speaker embeddings
# ---------------------------------------------------
class FaceToSpeakerNet(nn.Module):
    def __init__(self, frozen_face_model=casia_model, face_emb_dim=512, speaker_emb_dim=256):
        super(FaceToSpeakerNet, self).__init__()
        self.frozen_face_model = frozen_face_model
        
        # A deeper MLP with several layers:
        self.transform = nn.Sequential(
            nn.Linear(face_emb_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, speaker_emb_dim)
        )

    def forward(self, x, input_range='0-255'):
        """
        x: face images of shape [batch_size, 448, 448, 3]
        input_range: '0-255' or '0-1' indicating the range of input images
        """
        # 1. Permute the input tensor to [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)  # [batch_size, 3, 448, 448]
        
        # 2. Convert to float and normalize based on input range
        if input_range == '0-255':
            x = x.float() / 255.0  # Normalize to [0, 1]
        elif input_range == '0-1':
            x = x.float()  # Assume already in [0, 1]
        else:
            raise ValueError("input_range must be either '0-255' or '0-1'")

        # 3. Resize images from 448x448 to 160x160
        x_resized = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        
        # 4. Normalize the images to [-1, 1]
        x_normalized = (x_resized - 0.5) / 0.5  # Scale to [-1, 1]
        
        # 5. Generate face embedding from the frozen model
        with torch.no_grad():
            face_emb = self.frozen_face_model(x_normalized)  # [batch_size, 512]

        # 6. Transform face embedding to 256-D speaker embedding
        speaker_emb = self.transform(face_emb)   # [batch_size, 256]
        return speaker_emb

if __name__ == "__main__":
    # Instantiate the Face->Speaker model
    face_to_speaker_net = FaceToSpeakerNet(
        frozen_face_model=casia_model,
        face_emb_dim=512, 
        speaker_emb_dim=256
    ).to(device)  # Move model to the selected device

    # ---------------------------------------------------
    # 5. Define Loss Function and Optimizer
    # ---------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(face_to_speaker_net.transform.parameters(), lr=1e-4)

    # ---------------------------------------------------
    # 6. Training Loop with Dummy Variables
    # ---------------------------------------------------
    num_epochs = 10
    batch_size = 8

    face_to_speaker_net.train()  # Set to training mode

    for epoch in range(num_epochs):
        # a. Generate a batch of dummy face images
        #    - Shape: [batch_size, 448, 448, 3]
        #    - Values: Random integers in [0, 255] to simulate image pixels
        face_images = torch.randint(
            low=0, high=256, size=(batch_size, 448, 448, 3), dtype=torch.float32
        ).to(device)

        # b. Generate dummy target speaker embeddings
        #    - Shape: [batch_size, 256]
        #    - Values: Random floats
        target_speaker_emb = torch.randn(batch_size, 256).to(device)

        # c. Forward pass: face -> speaker
        predicted_speaker_emb = face_to_speaker_net(face_images, input_range='0-255')  # [batch_size, 256]

        # d. Compute loss
        loss = criterion(predicted_speaker_emb, target_speaker_emb)

        # e. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # f. Logging
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("Training completed.")
