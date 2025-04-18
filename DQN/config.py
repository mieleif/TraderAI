# In config.py
class TrainingConfig:
    EPISODES = 30
    BATCH_SIZE = 64
    STATE_SIZE = 10  # Updated to include funding rate
    ACTION_SIZE = 2
    
    # Common parameters
    GAMMA = 0.95
    LEARNING_RATE = 0.001
    MEMORY_SIZE = 5000
    
    # DQN specific parameters
    EPSILON = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    
    # PPO specific parameters
    PPO_CLIP_RATIO = 0.1
    PPO_POLICY_LR = 0.0001
    PPO_VALUE_LR = 0.0003
    PPO_ENTROPY_BETA = 0.005
    PPO_LAMBDA = 0.95
    PPO_EPOCHS = 10
    PPO_TRAJECTORIES = 2000
    
    # Continuous PPO parameters
    PPO_CONTINUOUS = True
    PPO_MIN_POSITION_SIZE = 0.1  # Minimum position size (10% of max)
    PPO_MAX_POSITION_SIZE = 0.5  # Maximum position size (100% of max)
    
    # Environment parameters
    HOLDING_PERIOD = 60
    RISK_PCT = 0.02
    POSITION_VALUE = 1000
    TIME_FRAME = "4h"

    # Add to TrainingConfig
    REWARD_SCALING = 1.0
    REWARD_SHAPING = True
    USE_DRAWDOWN_PENALTY = True
    TARGET_KL = 0.01

class DataConfig:
    REQUIRED_COLUMNS = [
        'open',
        'tenkan_sen',
        'kijun_sen',
        'senkou_span_a',
        'senkou_span_b',
        'close',
        'fundingRate',
        'ma_7',
        'ma_25',
        'ma_100'
    ]
    
    # Map of which columns are used in the state space
    STATE_COLUMNS = [
        'open',
        'tenkan_sen',
        'kijun_sen', 
        'senkou_span_a',
        'senkou_span_b',
        'close',
        'fundingRate'  # Explicitly include funding rate
        'ma_7',
        'ma_25',
        'ma_100'
    ]