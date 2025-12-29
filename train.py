import torch
from torch.utils.data import DataLoader, TensorDataset
from data_utils import load_data, prepare_data
from model import BiLSTMAttention

def train():
    df = load_data("GOOG")
    X, y, _ = prepare_data(df)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = BiLSTMAttention()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(30):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/30 | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "bilstm_model.pth")
    print("✅ Model saved successfully")

if __name__ == "__main__":
    train()
