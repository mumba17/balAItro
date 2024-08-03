import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageGrab
from collections import namedtuple, deque
from virtual_controller import VirtualController
from cnn_classifier import CNN
from reward_predictor import RewardPredictor
import keyboard


# Set up constants
MAX_STEPS_PER_EPISODE = 100
ORIGINAL_WIDTH = 240
ORIGINAL_HEIGHT = 135
SCALE_FACTOR = 1 # Adjust this to change the input size
INPUT_WIDTH = int(ORIGINAL_WIDTH * SCALE_FACTOR)
INPUT_HEIGHT = int(ORIGINAL_HEIGHT * SCALE_FACTOR)
NUM_ACTIONS = 12  # Adjust this based on the number of possible actions in your game
CONTROLLER = VirtualController()
PREDICTOR = RewardPredictor("vision-ss/models/cnn_classifier.pth")
previous_prediction = None
negative_reward_count = 0

print(f"Input size: {INPUT_WIDTH}x{INPUT_HEIGHT}")

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.head(x)

# Define a named tuple for our Transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Define the Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Function to capture screen
def capture_screen():
    screen = ImageGrab.grab().convert('L')  # Convert to grayscale
    screen = screen.resize((INPUT_WIDTH, INPUT_HEIGHT))
    screen = np.array(screen)
    screen = torch.from_numpy(screen).unsqueeze(0).unsqueeze(0).float() / 255
    return screen

# Function to select action
def select_action(state, policy_net, eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(NUM_ACTIONS)]], dtype=torch.long)

def step(action, current_step):
    global previous_prediction, negative_reward_count

    # Perform the action using the VirtualController
    button = CONTROLLER.num_to_act(action.item())
    if button is not None:
        CONTROLLER.press_button(button)
    
    # Capture new state
    next_state = capture_screen()
    
    # Get prediction from the CNN
    prediction = PREDICTOR.predict(next_state.squeeze(0).squeeze(0).numpy())
    
    # Calculate reward
    if prediction == 0:
        reward = 200
        negative_reward_count = 0

    elif prediction == 6:
        # If the prediction is 6, the model is entering the settings, which should be avoided
        reward = -1000
        done = True
        print("Episode ended: Entered settings (class 6)")
        keyboard.press_and_release('esc')
        time.sleep(2)
        keyboard.press('r')
        time.sleep(1.5)
        keyboard.release('r')
        time.sleep(5)
    elif previous_prediction == 2 and prediction == 1:
        reward = 5
        negative_reward_count = 0
    elif previous_prediction is not None and prediction != previous_prediction and prediction != 6:
        reward = -0.1 * negative_reward_count
        negative_reward_count = 0
    else:
        reward = -0.5 * negative_reward_count
        negative_reward_count += 1
        
    done = False
    
    # Update previous prediction
    previous_prediction = prediction
    
    reward = torch.tensor([reward], dtype=torch.float)

    if current_step >= MAX_STEPS_PER_EPISODE:
        print(f"Episode ended: Max steps reached ({MAX_STEPS_PER_EPISODE})")
        done = True
    elif reward.item() == 50:
        print("Episode ended: Reached reward stage (class 0)")
        done = True
    
    if done:
        print(f"Negative reward count: {negative_reward_count}")
    
    return next_state, reward, done


# Optimization function
def optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Training loop
def train():
    global negative_reward_count
    num_episodes = 1000
    batch_size = 128
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 600
    target_update = 2
    memory_size = 10000
    
    print("Starting training...")

    policy_net = DQN(INPUT_HEIGHT, INPUT_WIDTH, NUM_ACTIONS)
    
    print("Model initialized")
    
    target_net = DQN(INPUT_HEIGHT, INPUT_WIDTH, NUM_ACTIONS)
    
    print("Target model initialized")
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(memory_size)

    steps_done = 0

    for episode in range(num_episodes):
        print(f"Starting episode {episode}")
        state = capture_screen()
        negative_reward_count = 0
        for t in range(MAX_STEPS_PER_EPISODE):
            eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
            print(f"Step {t}, Epsilon: {eps_threshold}")
            action = select_action(state, policy_net, eps_threshold)
            next_state, reward, done = step(action, t)
            
            print(f"Action: {action.item()}, Reward: {reward.item()}")
            
            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma)
            steps_done += 1
            
            if done:
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode} completed")
        CONTROLLER.press_button(CONTROLLER.DPAD_DOWN)
        CONTROLLER.press_button(CONTROLLER.A)
        CONTROLLER.press_button(CONTROLLER.A)
        time.sleep(1)
        CONTROLLER.press_button(CONTROLLER.X)
        keyboard.press('r')
        
# Main execution
if __name__ == "__main__":
    train()