import socket
import threading
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict, deque
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import atexit
from datetime import datetime
import struct
import sys
import math
import glob
import shutil
import signal
import argparse
import traceback
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

# ==============================
# MODE CONFIGURATION - Set your preferred mode here
# ==============================
OPERATION_MODE = "training"  # Change to "production" for inference mode
# Available modes: "training", "production"

# ==============================
# TensorBoard Import Protection
# ==============================
try:
    from torch.utils.tensorboard import SummaryWriter
    from torch.distributions import Normal
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
    print("Warning: TensorBoard not available. Install tensorboard for logging support.")

# ==============================
# Enhanced Configuration System with Aggressive Exploration
# ==============================

@dataclass
class EnhancedSACConfig:
    """Enhanced SAC configuration with aggressive exploration settings"""
    # Core SAC parameters
    buffer_size: int = 100000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 0.0003
    lr_critic: float = 0.0003
    hidden_units: int = 256
    
    # AGGRESSIVE EXPLORATION PARAMETERS - OPTIMIZED FOR SHORT TRAINING
    initial_exploration_factor: float = 5.0        # Much more aggressive initial exploration
    exploration_decay: float = 0.995               # Faster decay for short training periods
    min_exploration: float = 1.0                   # Higher minimum exploration
    exploration_noise_scale: float = 1.0           # Much larger noise scale
    
    # NEW: Epsilon-greedy exploration for aggressive power exploration
    initial_epsilon: float = 0.8                   # High initial random action probability
    epsilon_decay: float = 0.992                   # Epsilon decay rate
    min_epsilon: float = 0.1                       # Minimum epsilon for continued exploration
    
    # NEW: Power-specific exploration parameters
    power_exploration_bonus: float = 2.0           # Extra exploration for power dimension
    power_random_prob: float = 0.3                 # Probability of random power selection
    power_random_decay: float = 0.995              # Decay for random power probability
    
    # NEW: Action noise parameters for better exploration
    action_noise_std: float = 0.2                  # Standard deviation for action noise
    action_noise_decay: float = 0.998              # Decay rate for action noise
    min_action_noise: float = 0.05                 # Minimum action noise
    
    # Enhanced reward parameters
    cbr_target: float = 0.65
    cbr_range: tuple = (0.6, 0.7)
    w1_cbr: float = 10.0      # CBR weight
    w2_power: float = 2.0     # Power penalty weight
    w3_sinr: float = 8.0      # SINR weight
    w4_mcs: float = 1.0       # MCS penalty weight
    beta: float = 20          # CBR response sharpness
    
    # Performance thresholds
    sinr_target: float = 15.0
    good_performance_cbr: float = 0.7
    good_performance_sinr: float = 12.0

@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Action bounds
    power_min: int = 1
    power_max: int = 30
    mcs_min: int = 0
    mcs_max: int = 10
    
    # Network settings
    host: str = '127.0.0.1'
    port: int = 5011
    
    # Reporting and saving
    log_dir: str = 'enhanced_sac_logs'
    model_save_dir: str = 'enhanced_sac_models'
    model_save_interval: int = 500  # requests
    report_interval: int = 1000     # requests
    auto_save_timeout: int = 10     # minutes
    periodic_save_interval: int = 5 # minutes
    
    # Logging
    log_received_path: str = 'received_data.log'
    log_sent_path: str = 'sent_data.log'
    log_metrics_path: str = 'metrics.log'

# Global configurations
sac_config = EnhancedSACConfig()
system_config = SystemConfig()

# Control flags
ENABLE_TENSORBOARD = TENSORBOARD_AVAILABLE
ENABLE_EXCEL_REPORTING = True

# Ensure directories exist
os.makedirs(system_config.log_dir, exist_ok=True)
if OPERATION_MODE == "training":
    if os.path.exists(system_config.model_save_dir):
        shutil.rmtree(system_config.model_save_dir)
    os.makedirs(system_config.model_save_dir, exist_ok=True)
else:
    os.makedirs(system_config.model_save_dir, exist_ok=True)

# ==============================
# Enhanced Logging Setup
# ==============================

def setup_enhanced_logging():
    """Setup comprehensive logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('enhanced_conventional_sac.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_enhanced_logging()

def log_data(log_path, data):
    """Log data to a file with a timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, 'a') as f:
        f.write(f"[{timestamp}] {data}\n")

# ==============================
# Model Loading Utilities
# ==============================

def find_latest_model_file(model_dir: str = system_config.model_save_dir) -> str:
    """Find the latest saved SAC model file"""
    try:
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory {model_dir} does not exist")
            return None
        
        # Look for SAC model files
        pattern = os.path.join(model_dir, "enhanced_sac_model_*.pth")
        model_files = glob.glob(pattern)
        
        if not model_files:
            # Fallback to original naming
            pattern = os.path.join(model_dir, "sac_model_*.pth")
            model_files = glob.glob(pattern)
        
        if not model_files:
            logger.warning("No saved model files found")
            return None
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_file = model_files[0]
        
        logger.info(f"Found latest model file: {latest_file}")
        return latest_file
        
    except Exception as e:
        logger.error(f"Error finding latest model file: {e}")
        return None

def validate_model_file(model_path: str) -> bool:
    """Validate that the model file exists and is loadable"""
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist: {model_path}")
            return False
        
        # Try to load the checkpoint to validate it
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check for required keys
        required_keys = [
            'policy_state_dict',
            'critic1_state_dict', 
            'critic2_state_dict',
            'critic1_target_state_dict',
            'critic2_target_state_dict',
            'log_alpha'
        ]
        
        for key in required_keys:
            if key not in checkpoint:
                logger.error(f"Required key missing in model file: {key}")
                return False
        
        logger.info("Model file validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Model file validation failed: {e}")
        return False

def load_latest_model_for_production():
    """Load the latest model automatically for production mode"""
    try:
        logger.info("PRODUCTION MODE: Loading latest saved model...")
        
        # Find latest model file
        latest_model = find_latest_model_file()
        
        if latest_model is None:
            logger.warning("No saved models found. Starting with fresh model in production mode.")
            logger.warning("This may result in suboptimal performance. Consider running training mode first.")
            return None
        
        # Validate model file
        if not validate_model_file(latest_model):
            logger.error("Latest model file is invalid. Starting with fresh model.")
            return None
        
        logger.info(f"PRODUCTION MODE: Successfully found model: {latest_model}")
        return latest_model
        
    except Exception as e:
        logger.error(f"Error loading model for production mode: {e}")
        return None

# ==============================
# Enhanced Excel Reporting System
# ==============================

class EnhancedSACExcelReporter:
    """Comprehensive Excel reporter for SAC performance analysis with exploration tracking"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Basic tracking
        self.request_count = 0
        self.start_time = time.time()
        self.vehicle_count = 0
        self.unique_vehicles = set()
        
        # Performance metrics
        self.total_rewards = []
        self.cbr_values = []
        self.sinr_values = []
        self.power_values = []
        self.mcs_values = []
        
        # NEW: Exploration tracking
        self.exploration_factors = []
        self.epsilon_values = []
        self.power_random_probs = []
        self.action_noise_values = []
        self.random_action_counts = []
        self.exploration_action_counts = []
        
        # SAC-specific metrics
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_values = []
        
        # Action tracking with enhanced power analysis
        self.power_actions = []
        self.mcs_actions = []
        self.action_distribution = defaultdict(int)
        self.power_distribution = defaultdict(int)  # NEW: Separate power distribution tracking
        
        # Performance episodes
        self.performance_episodes = []
        
        logger.info(f"Enhanced SAC Excel reporter with exploration tracking initialized")
    
    def add_batch_data(self, batch_data: Dict, batch_responses: Dict, 
                      performance_metrics: Dict = None, exploration_metrics: Dict = None):
        """Add batch data for comprehensive analysis including exploration metrics"""
        self.request_count += 1
        
        for vehicle_id, vehicle_data in batch_data.items():
            self.unique_vehicles.add(vehicle_id)
            self.vehicle_count += 1
            
            # Basic environment data
            cbr = float(vehicle_data.get('CBR', 0))
            sinr = float(vehicle_data.get('SNR', 0))
            power = float(vehicle_data.get('transmissionPower', 20))
            mcs = int(vehicle_data.get('MCS', 5))
            
            self.cbr_values.append(cbr)
            self.sinr_values.append(sinr)
            self.power_values.append(power)
            self.mcs_values.append(mcs)
            
            # Response data
            if vehicle_id in batch_responses.get('vehicles', {}):
                response = batch_responses['vehicles'][vehicle_id]
                
                # Training data
                if 'training' in response:
                    training_data = response['training']
                    self.total_rewards.append(training_data.get('reward', 0))
                    
                    # Action tracking
                    action = training_data.get('action', [0, 0])
                    if len(action) >= 2:
                        self.power_actions.append(action[0])
                        self.mcs_actions.append(action[1])
                        
                        # Action distribution
                        power_rounded = round(action[0], 1)
                        mcs_rounded = round(action[1])
                        self.action_distribution[(power_rounded, mcs_rounded)] += 1
                        
                        # NEW: Power distribution tracking
                        power_bin = int(action[0] / 2) * 2  # 2 dBm bins
                        self.power_distribution[power_bin] += 1
        
        # Performance metrics from SAC agent
        if performance_metrics:
            if 'actor_loss' in performance_metrics:
                self.actor_losses.append(performance_metrics['actor_loss'])
            if 'critic_loss' in performance_metrics:
                self.critic_losses.append(performance_metrics['critic_loss'])
            if 'alpha' in performance_metrics:
                self.alpha_values.append(performance_metrics['alpha'])
        
        # NEW: Exploration metrics tracking
        if exploration_metrics:
            self.exploration_factors.append(exploration_metrics.get('exploration_factor', 0))
            self.epsilon_values.append(exploration_metrics.get('epsilon', 0))
            self.power_random_probs.append(exploration_metrics.get('power_random_prob', 0))
            self.action_noise_values.append(exploration_metrics.get('action_noise_std', 0))
            self.random_action_counts.append(exploration_metrics.get('random_actions', 0))
            self.exploration_action_counts.append(exploration_metrics.get('exploration_actions', 0))
        
        # Record performance episode
        self.performance_episodes.append({
            'request': self.request_count,
            'timestamp': datetime.now(),
            'vehicles': len(batch_data),
            'avg_cbr': np.mean([float(v.get('CBR', 0)) for v in batch_data.values()]),
            'avg_sinr': np.mean([float(v.get('SNR', 0)) for v in batch_data.values()]),
            'avg_reward': np.mean(self.total_rewards[-len(batch_data):]) if self.total_rewards else 0,
            'avg_power': np.mean([float(v.get('transmissionPower', 20)) for v in batch_data.values()])
        })
    
    def generate_comprehensive_report(self, filename_prefix: str = "enhanced_sac_report") -> str:
        """Generate comprehensive Excel report with exploration analysis"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.xlsx"
            filepath = os.path.join(self.output_dir, filename)
            
            elapsed_time = time.time() - self.start_time
            
            # Summary statistics with exploration metrics
            summary_stats = {
                'Metric': [
                    'Total Requests Processed',
                    'Total Vehicle Interactions', 
                    'Unique Vehicles',
                    'Simulation Duration (minutes)',
                    'Requests per Minute',
                    'Average CBR',
                    'CBR Standard Deviation',
                    'CBR Target Achievement Rate (%)',
                    'Average SINR (dB)',
                    'SINR Standard Deviation',
                    'Good Performance Rate (%)',
                    'Average Reward',
                    'Reward Standard Deviation',
                    'Best Reward',
                    'Worst Reward',
                    'Average Power (dBm)',
                    'Power Standard Deviation',
                    'Power Range Explored (dBm)',
                    'Power Distribution Entropy',
                    'Average MCS',
                    'Action Diversity (unique actions)',
                    'Average Actor Loss',
                    'Average Critic Loss',
                    'Final Alpha Value',
                    'Final Exploration Factor',
                    'Final Epsilon Value',
                    'Final Power Random Probability',
                    'Total Random Actions',
                    'Total Exploration Actions'
                ],
                'Value': [
                    self.request_count,
                    self.vehicle_count,
                    len(self.unique_vehicles),
                    elapsed_time / 60,
                    (self.request_count / (elapsed_time / 60)) if elapsed_time > 0 else 0,
                    np.mean(self.cbr_values) if self.cbr_values else 0,
                    np.std(self.cbr_values) if self.cbr_values else 0,
                    self._calculate_cbr_achievement_rate(),
                    np.mean(self.sinr_values) if self.sinr_values else 0,
                    np.std(self.sinr_values) if self.sinr_values else 0,
                    self._calculate_good_performance_rate(),
                    np.mean(self.total_rewards) if self.total_rewards else 0,
                    np.std(self.total_rewards) if self.total_rewards else 0,
                    np.max(self.total_rewards) if self.total_rewards else 0,
                    np.min(self.total_rewards) if self.total_rewards else 0,
                    np.mean(self.power_values) if self.power_values else 0,
                    np.std(self.power_values) if self.power_values else 0,
                    (np.max(self.power_values) - np.min(self.power_values)) if self.power_values else 0,
                    self._calculate_power_entropy(),
                    np.mean(self.mcs_values) if self.mcs_values else 0,
                    len(self.action_distribution),
                    np.mean(self.actor_losses) if self.actor_losses else 0,
                    np.mean(self.critic_losses) if self.critic_losses else 0,
                    self.alpha_values[-1] if self.alpha_values else 0,
                    self.exploration_factors[-1] if self.exploration_factors else 0,
                    self.epsilon_values[-1] if self.epsilon_values else 0,
                    self.power_random_probs[-1] if self.power_random_probs else 0,
                    sum(self.random_action_counts),
                    sum(self.exploration_action_counts)
                ]
            }
            
            # Write comprehensive Excel with multiple sheets
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Performance over time
                if self.performance_episodes:
                    performance_df = pd.DataFrame(self.performance_episodes)
                    performance_df.to_excel(writer, sheet_name='Performance_Timeline', index=False)
                
                # Training metrics
                if self.actor_losses or self.critic_losses:
                    training_data = {
                        'Request': range(1, max(len(self.actor_losses), len(self.critic_losses)) + 1),
                        'Actor_Loss': self.actor_losses + [np.nan] * (max(len(self.actor_losses), len(self.critic_losses)) - len(self.actor_losses)),
                        'Critic_Loss': self.critic_losses + [np.nan] * (max(len(self.actor_losses), len(self.critic_losses)) - len(self.critic_losses)),
                        'Alpha': self.alpha_values + [np.nan] * (max(len(self.actor_losses), len(self.critic_losses)) - len(self.alpha_values))
                    }
                    training_df = pd.DataFrame(training_data)
                    training_df.to_excel(writer, sheet_name='Training_Metrics', index=False)
                
                # NEW: Exploration metrics sheet
                if self.exploration_factors:
                    exploration_data = {
                        'Request': range(1, len(self.exploration_factors) + 1),
                        'Exploration_Factor': self.exploration_factors,
                        'Epsilon': self.epsilon_values + [np.nan] * (len(self.exploration_factors) - len(self.epsilon_values)),
                        'Power_Random_Prob': self.power_random_probs + [np.nan] * (len(self.exploration_factors) - len(self.power_random_probs)),
                        'Action_Noise_Std': self.action_noise_values + [np.nan] * (len(self.exploration_factors) - len(self.action_noise_values)),
                        'Random_Actions': self.random_action_counts + [np.nan] * (len(self.exploration_factors) - len(self.random_action_counts)),
                        'Exploration_Actions': self.exploration_action_counts + [np.nan] * (len(self.exploration_factors) - len(self.exploration_action_counts))
                    }
                    exploration_df = pd.DataFrame(exploration_data)
                    exploration_df.to_excel(writer, sheet_name='Exploration_Metrics', index=False)
                
                # Action analysis
                if self.action_distribution:
                    action_analysis = []
                    for (power, mcs), count in self.action_distribution.items():
                        action_analysis.append({
                            'Power_Action': power,
                            'MCS_Action': mcs,
                            'Frequency': count,
                            'Percentage': (count / sum(self.action_distribution.values())) * 100
                        })
                    
                    action_df = pd.DataFrame(action_analysis)
                    action_df = action_df.sort_values('Frequency', ascending=False)
                    action_df.to_excel(writer, sheet_name='Action_Distribution', index=False)
                
                # NEW: Power distribution analysis
                if self.power_distribution:
                    power_analysis = []
                    for power_bin, count in self.power_distribution.items():
                        power_analysis.append({
                            'Power_Bin_Start': power_bin,
                            'Power_Bin_End': power_bin + 2,
                            'Frequency': count,
                            'Percentage': (count / sum(self.power_distribution.values())) * 100
                        })
                    
                    power_df = pd.DataFrame(power_analysis)
                    power_df = power_df.sort_values('Power_Bin_Start')
                    power_df.to_excel(writer, sheet_name='Power_Distribution', index=False)
                
                # Environment statistics
                if self.cbr_values and self.sinr_values:
                    env_stats = pd.DataFrame({
                        'Request': range(1, len(self.cbr_values) + 1),
                        'CBR': self.cbr_values,
                        'SINR': self.sinr_values,
                        'Power': self.power_values,
                        'MCS': self.mcs_values,
                        'Reward': self.total_rewards + [np.nan] * (len(self.cbr_values) - len(self.total_rewards))
                    })
                    env_stats.to_excel(writer, sheet_name='Environment_Data', index=False)
            
            logger.info(f"Enhanced SAC report with exploration analysis generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating enhanced report: {e}")
            return ""
    
    def _calculate_cbr_achievement_rate(self) -> float:
        """Calculate CBR target achievement rate"""
        if not self.cbr_values:
            return 0.0
        
        target_achievements = sum(1 for cbr in self.cbr_values 
                                if sac_config.cbr_range[0] <= cbr <= sac_config.cbr_range[1])
        return (target_achievements / len(self.cbr_values)) * 100
    
    def _calculate_good_performance_rate(self) -> float:
        """Calculate rate of good performance episodes"""
        if not self.cbr_values or not self.sinr_values:
            return 0.0
        
        good_episodes = 0
        for cbr, sinr in zip(self.cbr_values, self.sinr_values):
            if (cbr <= sac_config.good_performance_cbr and 
                sinr >= sac_config.good_performance_sinr):
                good_episodes += 1
        
        return (good_episodes / len(self.cbr_values)) * 100
    
    def _calculate_power_entropy(self) -> float:
        """Calculate entropy of power distribution to measure exploration diversity"""
        if not self.power_distribution:
            return 0.0
        
        total = sum(self.power_distribution.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in self.power_distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy

# ==============================
# Enhanced Neural Network Architectures
# ==============================

class EnhancedQNetwork(nn.Module):
    """Enhanced Q-Network with better initialization and normalization"""
    def __init__(self, state_dim, action_dim):
        super(EnhancedQNetwork, self).__init__()
        
        # Enhanced normalization
        self.register_buffer('state_mean', torch.tensor([0.65, 15.0, 5.0]))
        self.register_buffer('state_std', torch.tensor([0.2, 5.0, 3.0]))
        
        self.fc1 = nn.Linear(state_dim + action_dim, sac_config.hidden_units)
        self.fc2 = nn.Linear(sac_config.hidden_units, sac_config.hidden_units)
        self.fc3 = nn.Linear(sac_config.hidden_units, sac_config.hidden_units // 2)
        self.fc4 = nn.Linear(sac_config.hidden_units // 2, 1)
        
        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        """Enhanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, state, action):
        # Normalize state
        if state.shape[-1] == 3:
            state_norm = (state - self.state_mean) / (self.state_std + 1e-8)
            state_norm = torch.clamp(state_norm, -5.0, 5.0)
        else:
            state_norm = state
            
        x = torch.cat([state_norm, action], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class EnhancedGaussianPolicy(nn.Module):
    """Enhanced Gaussian Policy with better exploration control"""
    def __init__(self, state_dim, action_dim):
        super(EnhancedGaussianPolicy, self).__init__()
        
        # Enhanced normalization
        self.register_buffer('state_mean', torch.tensor([0.65, 15.0, 5.0]))
        self.register_buffer('state_std', torch.tensor([0.2, 5.0, 3.0]))
        
        self.fc1 = nn.Linear(state_dim, sac_config.hidden_units)
        self.fc2 = nn.Linear(sac_config.hidden_units, sac_config.hidden_units)
        self.fc3 = nn.Linear(sac_config.hidden_units, sac_config.hidden_units // 2)
        
        self.mean = nn.Linear(sac_config.hidden_units // 2, action_dim)
        self.log_std = nn.Linear(sac_config.hidden_units // 2, action_dim)
        
        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        """Enhanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m == self.mean:
                    nn.init.uniform_(m.weight.data, -0.003, 0.003)
                    nn.init.uniform_(m.bias.data, -0.003, 0.003)
                else:
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0)

    def forward(self, state):
        # Normalize state
        if state.shape[-1] == 3:
            state_norm = (state - self.state_mean) / (self.state_std + 1e-8)
            state_norm = torch.clamp(state_norm, -5.0, 5.0)
        else:
            state_norm = state
            
        x = F.relu(self.fc1(state_norm))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if TENSORBOARD_AVAILABLE:
            normal = Normal(mean, std)
            x_t = normal.rsample()
        else:
            x_t = mean + torch.randn_like(mean) * std
            
        action = torch.tanh(x_t)
        
        # Calculate log probability
        if TENSORBOARD_AVAILABLE:
            log_prob = normal.log_prob(x_t)
        else:
            log_prob = -0.5 * (((x_t - mean) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi))
            
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# ==============================
# Enhanced SAC Agent with Aggressive Exploration
# ==============================

class EnhancedSACAgent:
    def __init__(self, state_dim, action_dim, training_mode=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training_mode = training_mode
        
        # Enhanced networks
        self.critic1 = EnhancedQNetwork(state_dim, action_dim)
        self.critic2 = EnhancedQNetwork(state_dim, action_dim)
        self.critic1_target = EnhancedQNetwork(state_dim, action_dim)
        self.critic2_target = EnhancedQNetwork(state_dim, action_dim)
        self.policy = EnhancedGaussianPolicy(state_dim, action_dim)
        
        # Copy targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Enhanced optimizers (only needed for training)
        if training_mode:
            self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=sac_config.lr_critic)
            self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=sac_config.lr_critic)
            self.policy_optim = optim.Adam(self.policy.parameters(), lr=sac_config.lr_actor)
            
            # Automatic entropy tuning
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=sac_config.lr_actor)
            
            # Enhanced replay buffer
            self.replay_buffer = deque(maxlen=sac_config.buffer_size)
        else:
            # Production mode - minimal setup
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=False)
            self.replay_buffer = None
            logger.info("SAC Agent initialized in PRODUCTION mode (no training)")
        
        # Enhanced metrics and tracking
        self.cumulative_rewards = []
        self.action_distribution = defaultdict(int)
        self.performance_metrics = {
            'actor_losses': [],
            'critic1_losses': [],
            'critic2_losses': [],
            'alpha_losses': [],
            'alpha_values': [],
            'q_values': [],
            'policy_entropy': []
        }
        
        # AGGRESSIVE EXPLORATION PARAMETERS
        self.exploration_factor = sac_config.initial_exploration_factor
        self.epsilon = sac_config.initial_epsilon  # NEW: Epsilon-greedy exploration
        self.power_random_prob = sac_config.power_random_prob  # NEW: Power-specific random probability
        self.action_noise_std = sac_config.action_noise_std  # NEW: Action noise standard deviation
        
        # Exploration tracking
        self.random_action_count = 0
        self.exploration_action_count = 0
        self.policy_action_count = 0
        
        self.training_steps = 0
        self.last_save_time = time.time()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # TensorBoard writer
        if training_mode and ENABLE_TENSORBOARD and TENSORBOARD_AVAILABLE:
            try:
                log_dir = os.path.join(system_config.log_dir, f"sac_tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir)
                logger.info(f"TensorBoard logging enabled: {log_dir}")
            except Exception as e:
                logger.warning(f"TensorBoard setup failed: {e}")
                self.writer = None
        else:
            self.writer = None
        
        # Load model automatically in production mode
        if not training_mode:
            self._load_latest_model_auto()
        
        # Log initial exploration parameters
        logger.info("AGGRESSIVE EXPLORATION PARAMETERS:")
        logger.info(f"  Initial exploration factor: {self.exploration_factor}")
        logger.info(f"  Initial epsilon: {self.epsilon}")
        logger.info(f"  Power random probability: {self.power_random_prob}")
        logger.info(f"  Action noise std: {self.action_noise_std}")
    
    def _load_latest_model_auto(self):
        """Automatically load latest model for production mode"""
        try:
            latest_model_path = load_latest_model_for_production()
            if latest_model_path:
                self.load_model(latest_model_path)
                logger.info("PRODUCTION MODE: Pre-trained model loaded successfully")
                logger.info(f"Model source: {latest_model_path}")
                logger.info("Ready for optimized inference")
            else:
                logger.warning("PRODUCTION MODE: No valid model found, using random initialization")
                logger.warning("Performance may be suboptimal")
        except Exception as e:
            logger.error(f"Failed to auto-load model in production mode: {e}")
            logger.warning("Using random initialization")

    def select_action(self, state, training=None):
        """ENHANCED action selection with AGGRESSIVE exploration strategies"""
        if training is None:
            training = self.training_mode
            
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.training_mode and training:
                # STRATEGY 1: Epsilon-greedy for complete random exploration
                if random.random() < self.epsilon:
                    # Complete random action
                    power = random.uniform(system_config.power_min, system_config.power_max)
                    mcs = random.randint(system_config.mcs_min, system_config.mcs_max)
                    self.random_action_count += 1
                    
                    logger.debug(f"RANDOM ACTION: Power={power:.1f}, MCS={mcs}, Epsilon={self.epsilon:.3f}")
                    return np.array([power, mcs], dtype=np.float32)
                
                # STRATEGY 2: Power-specific random exploration
                if random.random() < self.power_random_prob:
                    # Get MCS from policy, randomize power aggressively
                    mean, _ = self.policy.forward(state)
                    mcs_action = torch.tanh(mean[0, 1])
                    mcs = self.denormalize(mcs_action.item(), system_config.mcs_min, system_config.mcs_max)
                    mcs = int(max(system_config.mcs_min, min(system_config.mcs_max, mcs)))
                    
                    # Aggressive power randomization
                    power = random.uniform(system_config.power_min, system_config.power_max)
                    self.exploration_action_count += 1
                    
                    logger.debug(f"POWER RANDOM ACTION: Power={power:.1f} (random), MCS={mcs} (policy)")
                    return np.array([power, mcs], dtype=np.float32)
                
                # STRATEGY 3: Enhanced policy sampling with aggressive noise
                action, log_prob = self.policy.sample(state)
                
                # Apply multiple layers of exploration noise
                if self.exploration_factor > 1.0:
                    # Layer 1: General exploration noise
                    exploration_noise = torch.randn_like(action) * (self.exploration_factor - 1.0) * sac_config.exploration_noise_scale
                    action = action + exploration_noise
                    
                    # Layer 2: Power-specific aggressive noise
                    power_noise = torch.randn(1) * sac_config.power_exploration_bonus * sac_config.exploration_noise_scale
                    action[0, 0] = action[0, 0] + power_noise
                    
                    # Layer 3: Additional action noise
                    action_noise = torch.randn_like(action) * self.action_noise_std
                    action = action + action_noise
                    
                    # Keep in valid range
                    action = torch.clamp(action, -1.0, 1.0)
                
                # Update exploration parameters more aggressively
                self.exploration_factor *= sac_config.exploration_decay
                self.exploration_factor = max(self.exploration_factor, sac_config.min_exploration)
                
                self.epsilon *= sac_config.epsilon_decay
                self.epsilon = max(self.epsilon, sac_config.min_epsilon)
                
                self.power_random_prob *= sac_config.power_random_decay
                self.power_random_prob = max(self.power_random_prob, 0.05)  # Keep some randomness
                
                self.action_noise_std *= sac_config.action_noise_decay
                self.action_noise_std = max(self.action_noise_std, sac_config.min_action_noise)
                
                self.policy_action_count += 1
                
            else:
                # Production mode or deterministic: use mean of policy
                mean, _ = self.policy.forward(state)
                action = torch.tanh(mean)  # Deterministic action
        
        # Denormalize actions
        power = self.denormalize(action[0, 0].item(), system_config.power_min, system_config.power_max)
        mcs = self.denormalize(action[0, 1].item(), system_config.mcs_min, system_config.mcs_max)
        
        # Apply bounds
        power = max(system_config.power_min, min(system_config.power_max, power))
        mcs = int(max(system_config.mcs_min, min(system_config.mcs_max, mcs)))
        
        # Track action distribution
        self.action_distribution[(round(power, 1), mcs)] += 1
        
        return np.array([power, mcs], dtype=np.float32)

    def denormalize(self, action, low, high):
        """Denormalize action from [-1, 1] to [low, high]"""
        return low + (action + 1) * (high - low) / 2

    def calculate_enhanced_reward(self, state, action, next_state):
        """Enhanced reward calculation with comprehensive metrics"""
        try:
            cbr, sinr, _ = next_state
            power, mcs = action
            
            # CBR reward component (most important)
            cbr_error = abs(cbr - sac_config.cbr_target)
            if cbr_error <= 0.025:
                cbr_reward = sac_config.w1_cbr * 2.0  # Double reward for excellent CBR
            elif cbr_error <= 0.05:
                cbr_reward = sac_config.w1_cbr * 1.0
            else:
                cbr_reward = sac_config.w1_cbr * (1 - math.tanh(sac_config.beta * cbr_error))
            
            # Power efficiency reward
            power_norm = (power - system_config.power_min) / (system_config.power_max - system_config.power_min)
            power_penalty = sac_config.w2_power * (power_norm ** 2)
            
            # SINR reward component
            sinr_diff = sinr - sac_config.sinr_target
            if abs(sinr_diff) < 2.0:
                sinr_reward = sac_config.w3_sinr * 0.5  # Small reward for good SINR
            else:
                sinr_reward = sac_config.w3_sinr * math.tanh(sinr_diff / 10.0)
            
            # MCS efficiency reward
            mcs_norm = (mcs - system_config.mcs_min) / (system_config.mcs_max - system_config.mcs_min)
            mcs_reward = sac_config.w4_mcs * (1 - abs(mcs_norm - 0.5))
            
            # Combined reward
            total_reward = cbr_reward - power_penalty + sinr_reward + mcs_reward
            
            # Bonus for excellent performance
            if cbr <= sac_config.good_performance_cbr and sinr >= sac_config.good_performance_sinr:
                total_reward += 5.0  # Bonus for excellent performance
            
            return float(total_reward)
            
        except Exception as e:
            logger.error(f"Enhanced reward calculation failed: {e}")
            return 0.0

    def update(self):
        """Enhanced update with comprehensive metrics tracking"""
        if not self.training_mode:
            return {}  # Skip updates in production mode
            
        if len(self.replay_buffer) < sac_config.batch_size:
            return {}
            
        batch = random.sample(self.replay_buffer, sac_config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Update critics
        critic1_loss, critic2_loss = self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update actor and alpha
        actor_loss, alpha_loss, alpha_value = self._update_actor_and_alpha(states)
        
        # Update target networks
        self._soft_update_targets()
        
        self.training_steps += 1
        
        # Track metrics
        metrics = {
            'actor_loss': float(actor_loss),
            'critic1_loss': float(critic1_loss),
            'critic2_loss': float(critic2_loss),
            'alpha_loss': float(alpha_loss),
            'alpha': float(alpha_value),
            'exploration_factor': float(self.exploration_factor),
            'epsilon': float(self.epsilon),
            'power_random_prob': float(self.power_random_prob),
            'action_noise_std': float(self.action_noise_std),
            'training_steps': int(self.training_steps),
            'random_actions': int(self.random_action_count),
            'exploration_actions': int(self.exploration_action_count),
            'policy_actions': int(self.policy_action_count)
        }
        
        # Store metrics
        self.performance_metrics['actor_losses'].append(actor_loss)
        self.performance_metrics['critic1_losses'].append(critic1_loss)
        self.performance_metrics['critic2_losses'].append(critic2_loss)
        self.performance_metrics['alpha_losses'].append(alpha_loss)
        self.performance_metrics['alpha_values'].append(alpha_value)
        
        # TensorBoard logging with exploration metrics
        if self.writer:
            self.writer.add_scalar('Loss/Actor', actor_loss, self.training_steps)
            self.writer.add_scalar('Loss/Critic1', critic1_loss, self.training_steps)
            self.writer.add_scalar('Loss/Critic2', critic2_loss, self.training_steps)
            self.writer.add_scalar('Loss/Alpha', alpha_loss, self.training_steps)
            self.writer.add_scalar('Parameters/Alpha', alpha_value, self.training_steps)
            self.writer.add_scalar('Exploration/Factor', self.exploration_factor, self.training_steps)
            self.writer.add_scalar('Exploration/Epsilon', self.epsilon, self.training_steps)
            self.writer.add_scalar('Exploration/Power_Random_Prob', self.power_random_prob, self.training_steps)
            self.writer.add_scalar('Exploration/Action_Noise_Std', self.action_noise_std, self.training_steps)
            self.writer.add_scalar('Actions/Random_Count', self.random_action_count, self.training_steps)
            self.writer.add_scalar('Actions/Exploration_Count', self.exploration_action_count, self.training_steps)
            self.writer.add_scalar('Actions/Policy_Count', self.policy_action_count, self.training_steps)
        
        return metrics

    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks"""
        # Calculate targets
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
            target_q = rewards + (1 - dones) * sac_config.gamma * q_next
        
        # Current Q values
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        # Losses
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        
        # Update Critic 1
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optim.step()
        
        # Update Critic 2
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optim.step()
        
        return critic1_loss.item(), critic2_loss.item()

    def _update_actor_and_alpha(self, states):
        """Update actor and alpha"""
        # Actor loss
        actions_pred, log_probs = self.policy.sample(states)
        q1_pred = self.critic1(states, actions_pred)
        q2_pred = self.critic2(states, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)
        
        actor_loss = (self.log_alpha.exp() * log_probs - q_pred).mean()
        
        # Update actor
        self.policy_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optim.step()
        
        # Alpha loss
        with torch.no_grad():
            _, log_probs_detached = self.policy.sample(states)
        alpha_loss = -(self.log_alpha * (log_probs_detached + self.target_entropy)).mean()
        
        # Update alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        
        alpha_value = self.log_alpha.exp().item()
        
        return actor_loss.item(), alpha_loss.item(), alpha_value

    def _soft_update_targets(self):
        """Soft update target networks"""
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(sac_config.tau * param.data + (1 - sac_config.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(sac_config.tau * param.data + (1 - sac_config.tau) * target_param.data)

    def save_model(self, path=None):
        """Enhanced model saving with comprehensive state including exploration parameters"""
        if not self.training_mode:
            logger.info("Model saving skipped in production mode")
            return
            
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(system_config.model_save_dir, f"enhanced_sac_model_{timestamp}.pth")
        
        # Also save to the main path for backward compatibility
        main_path = os.path.join(system_config.model_save_dir, "enhanced_sac_model.pth")
        
        # Comprehensive model data with exploration parameters
        model_data = {
            'policy_state_dict': self.policy.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'target_entropy': self.target_entropy,
            'exploration_factor': self.exploration_factor,
            'epsilon': self.epsilon,  # NEW
            'power_random_prob': self.power_random_prob,  # NEW
            'action_noise_std': self.action_noise_std,  # NEW
            'training_steps': self.training_steps,
            'cumulative_rewards': self.cumulative_rewards,
            'action_distribution': dict(self.action_distribution),
            'performance_metrics': self.performance_metrics,
            'exploration_counts': {  # NEW
                'random_actions': self.random_action_count,
                'exploration_actions': self.exploration_action_count,
                'policy_actions': self.policy_action_count
            },
            'config': {
                'sac_config': sac_config.__dict__.copy(),
                'system_config': system_config.__dict__.copy()
            },
            'mode': 'training',
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        # Save timestamped version
        torch.save(model_data, path)
        
        # Save main version
        torch.save(model_data, main_path)
        
        logger.info(f"Enhanced model with exploration parameters saved to {path} and {main_path}")
        logger.info(f"Training steps: {self.training_steps}, Exploration: {self.exploration_factor:.4f}")
        logger.info(f"Epsilon: {self.epsilon:.4f}, Power random prob: {self.power_random_prob:.4f}")
        self.last_save_time = time.time()

    def load_model(self, path):
        """Enhanced model loading with exploration parameters validation"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load network states
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
            self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            
            # Load training state
            self.target_entropy = checkpoint.get('target_entropy', self.target_entropy)
            self.exploration_factor = checkpoint.get('exploration_factor', sac_config.initial_exploration_factor)
            self.training_steps = checkpoint.get('training_steps', 0)
            
            # Load exploration parameters (NEW)
            self.epsilon = checkpoint.get('epsilon', sac_config.initial_epsilon)
            self.power_random_prob = checkpoint.get('power_random_prob', sac_config.power_random_prob)
            self.action_noise_std = checkpoint.get('action_noise_std', sac_config.action_noise_std)
            
            # Load exploration counts (NEW)
            exploration_counts = checkpoint.get('exploration_counts', {})
            self.random_action_count = exploration_counts.get('random_actions', 0)
            self.exploration_action_count = exploration_counts.get('exploration_actions', 0)
            self.policy_action_count = exploration_counts.get('policy_actions', 0)
            
            # Load metrics
            self.cumulative_rewards = checkpoint.get('cumulative_rewards', [])
            self.action_distribution = defaultdict(int, checkpoint.get('action_distribution', {}))
            self.performance_metrics = checkpoint.get('performance_metrics', {
                'actor_losses': [], 'critic1_losses': [], 'critic2_losses': [],
                'alpha_losses': [], 'alpha_values': [], 'q_values': [], 'policy_entropy': []
            })
            
            # Log comprehensive information
            model_mode = checkpoint.get('mode', 'unknown')
            model_timestamp = checkpoint.get('timestamp', 'unknown')
            model_device = checkpoint.get('device', 'unknown')
            
            logger.info(f"Enhanced model with exploration parameters loaded from {path}")
            logger.info(f"Model details:")
            logger.info(f"  - Saved from: {model_mode} mode")
            logger.info(f"  - Timestamp: {model_timestamp}")
            logger.info(f"  - Original device: {model_device}")
            logger.info(f"  - Training steps: {self.training_steps}")
            logger.info(f"  - Exploration factor: {self.exploration_factor:.4f}")
            logger.info(f"  - Epsilon: {self.epsilon:.4f}")
            logger.info(f"  - Power random prob: {self.power_random_prob:.4f}")
            logger.info(f"  - Action noise std: {self.action_noise_std:.4f}")
            logger.info(f"  - Alpha value: {self.log_alpha.exp().item():.4f}")
            logger.info(f"  - Action counts: Random={self.random_action_count}, Exploration={self.exploration_action_count}, Policy={self.policy_action_count}")
            logger.info(f"  - Cumulative rewards history: {len(self.cumulative_rewards)} episodes")
            logger.info(f"  - Action distribution entries: {len(self.action_distribution)}")
            
        except Exception as e:
            logger.error(f"Failed to load enhanced model from {path}: {e}")
            raise e

    def generate_performance_plots(self, save_dir=None):
        """Generate comprehensive performance plots including exploration metrics"""
        if not self.training_mode:
            logger.info("Performance plotting skipped in production mode")
            return
            
        if save_dir is None:
            save_dir = system_config.log_dir
        
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(3, 3, figsize=(20, 15))
            fig.suptitle('Enhanced SAC Performance Analysis with Aggressive Exploration', fontsize=16, fontweight='bold')
            
            # 1. Cumulative rewards
            if self.cumulative_rewards:
                axes[0, 0].plot(self.cumulative_rewards, alpha=0.7, linewidth=1.5)
                axes[0, 0].plot(self._smooth_curve(self.cumulative_rewards), color='red', linewidth=2, label='Smoothed')
                axes[0, 0].set_title('Cumulative Rewards')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Training losses
            if self.performance_metrics['actor_losses']:
                axes[0, 1].plot(self.performance_metrics['actor_losses'], label='Actor Loss', alpha=0.8)
                if self.performance_metrics['critic1_losses']:
                    axes[0, 1].plot(self.performance_metrics['critic1_losses'], label='Critic 1 Loss', alpha=0.8)
                if self.performance_metrics['critic2_losses']:
                    axes[0, 1].plot(self.performance_metrics['critic2_losses'], label='Critic 2 Loss', alpha=0.8)
                axes[0, 1].set_title('Training Losses')
                axes[0, 1].set_xlabel('Training Step')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_yscale('log')
            
            # 3. Alpha values
            if self.performance_metrics['alpha_values']:
                axes[0, 2].plot(self.performance_metrics['alpha_values'], color='green', linewidth=2)
                axes[0, 2].set_title('Alpha (Temperature) Evolution')
                axes[0, 2].set_xlabel('Training Step')
                axes[0, 2].set_ylabel('Alpha Value')
                axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Action distribution heatmap
            if self.action_distribution:
                power_actions = [k[0] for k in self.action_distribution.keys()]
                mcs_actions = [k[1] for k in self.action_distribution.keys()]
                counts = list(self.action_distribution.values())
                
                scatter = axes[1, 0].scatter(power_actions, mcs_actions, 
                                           s=[c/max(counts)*200 for c in counts],
                                           c=counts, cmap='viridis', alpha=0.7)
                axes[1, 0].set_title('Action Distribution')
                axes[1, 0].set_xlabel('Power Action (dBm)')
                axes[1, 0].set_ylabel('MCS Action')
                axes[1, 0].set_xlim(system_config.power_min-1, system_config.power_max+1)
                plt.colorbar(scatter, ax=axes[1, 0], label='Frequency')
            
            # 5. Power distribution histogram
            if self.action_distribution:
                power_values = [k[0] for k in self.action_distribution.keys()]
                power_counts = [self.action_distribution[k] for k in self.action_distribution.keys()]
                axes[1, 1].hist(power_values, weights=power_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, 1].set_title('Power Distribution')
                axes[1, 1].set_xlabel('Power (dBm)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_xlim(system_config.power_min, system_config.power_max)
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Exploration metrics over time (NEW)
            steps = range(len(self.performance_metrics['alpha_values']))
            if steps:
                axes[1, 2].plot(steps, [self.exploration_factor] * len(steps), label='Exploration Factor', alpha=0.8)
                # Note: These would need to be tracked over time in the actual implementation
                axes[1, 2].set_title('Exploration Parameters')
                axes[1, 2].set_xlabel('Training Step')
                axes[1, 2].set_ylabel('Value')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            
            # 7. Learning progress
            if len(self.cumulative_rewards) > 100:
                window_size = min(100, len(self.cumulative_rewards) // 10)
                rolling_mean = self._rolling_mean(self.cumulative_rewards, window_size)
                axes[2, 0].plot(rolling_mean, color='purple', linewidth=2)
                axes[2, 0].set_title(f'Learning Progress (Window: {window_size})')
                axes[2, 0].set_xlabel('Episode')
                axes[2, 0].set_ylabel('Average Reward')
                axes[2, 0].grid(True, alpha=0.3)
            
            # 8. Action type distribution (NEW)
            action_types = ['Random', 'Exploration', 'Policy']
            action_counts = [self.random_action_count, self.exploration_action_count, self.policy_action_count]
            if sum(action_counts) > 0:
                colors = ['red', 'orange', 'blue']
                axes[2, 1].pie(action_counts, labels=action_types, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[2, 1].set_title('Action Selection Strategy Distribution')
            
            # 9. Performance metrics summary
            axes[2, 2].axis('off')
            summary_text = f"""Performance Summary:
            
Training Steps: {self.training_steps:,}
Total Episodes: {len(self.cumulative_rewards):,}
Current Alpha: {self.log_alpha.exp().item():.4f}

Exploration Parameters:
Exploration Factor: {self.exploration_factor:.4f}
Epsilon: {self.epsilon:.4f}
Power Random Prob: {self.power_random_prob:.4f}
Action Noise Std: {self.action_noise_std:.4f}

Action Counts:
Random: {self.random_action_count:,}
Exploration: {self.exploration_action_count:,}
Policy: {self.policy_action_count:,}

Recent Performance:
Best Reward: {max(self.cumulative_rewards[-100:]) if len(self.cumulative_rewards) >= 100 else (max(self.cumulative_rewards) if self.cumulative_rewards else 0):.2f}
Avg Reward (last 100): {np.mean(self.cumulative_rewards[-100:]) if len(self.cumulative_rewards) >= 100 else (np.mean(self.cumulative_rewards) if self.cumulative_rewards else 0):.2f}

Power Exploration Range:
Min Power Used: {min([k[0] for k in self.action_distribution.keys()]) if self.action_distribution else 'N/A'}
Max Power Used: {max([k[0] for k in self.action_distribution.keys()]) if self.action_distribution else 'N/A'}
"""
            axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes,
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(save_dir, f'enhanced_sac_aggressive_exploration_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Enhanced performance plots with exploration analysis saved: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error generating performance plots: {e}")

    def _smooth_curve(self, data, weight=0.9):
        """Smooth curve using exponential moving average"""
        if not data:
            return []
        smoothed = [data[0]]
        for point in data[1:]:
            smoothed.append(weight * smoothed[-1] + (1 - weight) * point)
        return smoothed

    def _rolling_mean(self, data, window):
        """Calculate rolling mean"""
        if len(data) < window:
            return data
        return [np.mean(data[i:i+window]) for i in range(len(data) - window + 1)]

    def get_comprehensive_metrics(self):
        """Get comprehensive performance metrics including exploration"""
        try:
            metrics = {
                'training_steps': self.training_steps,
                'exploration_factor': self.exploration_factor,
                'epsilon': self.epsilon,
                'power_random_prob': self.power_random_prob,
                'action_noise_std': self.action_noise_std,
                'alpha_value': self.log_alpha.exp().item(),
                'total_episodes': len(self.cumulative_rewards),
                'unique_actions': len(self.action_distribution),
                'random_actions': self.random_action_count,
                'exploration_actions': self.exploration_action_count,
                'policy_actions': self.policy_action_count
            }
            
            if self.cumulative_rewards:
                metrics.update({
                    'avg_reward': np.mean(self.cumulative_rewards),
                    'best_reward': np.max(self.cumulative_rewards),
                    'worst_reward': np.min(self.cumulative_rewards),
                    'reward_std': np.std(self.cumulative_rewards),
                    'recent_avg_reward': np.mean(self.cumulative_rewards[-100:]) if len(self.cumulative_rewards) >= 100 else np.mean(self.cumulative_rewards)
                })
            
            if self.action_distribution:
                power_values = [k[0] for k in self.action_distribution.keys()]
                metrics.update({
                    'min_power_explored': min(power_values),
                    'max_power_explored': max(power_values),
                    'power_range_explored': max(power_values) - min(power_values)
                })
            
            if self.performance_metrics['actor_losses']:
                metrics.update({
                    'avg_actor_loss': np.mean(self.performance_metrics['actor_losses'][-100:]),
                    'avg_critic_loss': np.mean(self.performance_metrics['critic1_losses'][-100:]) if self.performance_metrics['critic1_losses'] else 0
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting comprehensive metrics: {e}")
            return {}

    def check_save_interval(self):
        """Check if it's time to save model"""
        if not self.training_mode:
            return
            
        current_time = time.time()
        if current_time - self.last_save_time >= (system_config.periodic_save_interval * 60):
            self.save_model()

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.writer:
                self.writer.close()
                logger.info("TensorBoard writer closed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# ==============================
# Enhanced Auto-Save Manager
# ==============================

class EnhancedAutoSaveManager:
    """Enhanced auto-save manager with comprehensive reporting"""
    
    def __init__(self, timeout_minutes: int):
        self.timeout_minutes = timeout_minutes
        self.last_activity = time.time()
        self.periodic_save_interval = system_config.periodic_save_interval * 60
        self.last_periodic_save = time.time()
        self.running = True
        self.save_callback = None
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_activity, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Enhanced auto-save manager started:")
        logger.info(f"  - Idle timeout: {timeout_minutes} minutes")
        logger.info(f"  - Periodic save interval: {system_config.periodic_save_interval} minutes")
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def set_save_callback(self, callback):
        """Set the callback function for saving"""
        self.save_callback = callback
    
    def _monitor_activity(self):
        """Monitor activity and trigger saves when needed"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check for idle timeout
                idle_time = current_time - self.last_activity
                if idle_time >= (self.timeout_minutes * 60):
                    logger.info(f"System idle for {idle_time/60:.1f} minutes, triggering auto-save...")
                    if self.save_callback:
                        self.save_callback("idle_timeout")
                    self.last_activity = current_time
                
                # Check for periodic save
                time_since_periodic = current_time - self.last_periodic_save
                if time_since_periodic >= self.periodic_save_interval:
                    logger.info(f"Periodic save triggered after {time_since_periodic/60:.1f} minutes")
                    if self.save_callback:
                        self.save_callback("periodic")
                    self.last_periodic_save = current_time
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Auto-save monitor error: {e}")
                time.sleep(60)
    
    def stop(self):
        """Stop the auto-save manager"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

# ==============================
# Enhanced RL Server
# ==============================

class EnhancedRLServer:
    def __init__(self, host, port, training_mode=True, timeout_minutes=None):
        self.host = host
        self.port = port
        self.training_mode = training_mode
        self.running = True
        
        # Request-based tracking
        self.request_count = 0
        
        # Enhanced auto-save management (only for training mode)
        if training_mode:
            timeout = timeout_minutes or system_config.auto_save_timeout
            self.auto_save_manager = EnhancedAutoSaveManager(timeout)
            self.auto_save_manager.set_save_callback(self._auto_save_callback)
        else:
            self.auto_save_manager = None
        
        # Enhanced Excel reporting
        if ENABLE_EXCEL_REPORTING:
            self.excel_reporter = EnhancedSACExcelReporter()
        else:
            self.excel_reporter = None
        
        # Enhanced SAC agent with aggressive exploration
        self.agent = EnhancedSACAgent(state_dim=3, action_dim=2, training_mode=training_mode)
        
        # Request-based intervals
        self.last_report_request = 0
        self.last_save_request = 0
        
        # Setup server socket
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        
        # Buffer configuration
        self.BUFFER_SIZE = 4096
        self.HEADER_SIZE = 4
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Enhanced SAC Server with Aggressive Exploration Configuration:")
        logger.info(f"  - Host: {host}:{port}")
        logger.info(f"  - Mode: {'TRAINING' if training_mode else 'PRODUCTION'}")
        logger.info(f"  - CBR Target: {sac_config.cbr_target}")
        logger.info(f"  - Power range: {system_config.power_min}-{system_config.power_max}")
        logger.info(f"  - MCS range: {system_config.mcs_min}-{system_config.mcs_max}")
        if training_mode:
            logger.info(f"  - Auto-save timeout: {timeout} minutes")
            logger.info(f"  - Initial exploration factor: {sac_config.initial_exploration_factor}")
            logger.info(f"  - Initial epsilon: {sac_config.initial_epsilon}")
            logger.info(f"  - Power random probability: {sac_config.power_random_prob}")

    def start(self):
        """Start the enhanced server"""
        logger.info(f"Starting Enhanced SAC server with AGGRESSIVE EXPLORATION in {'TRAINING' if self.training_mode else 'PRODUCTION'} mode")
        
        while self.running:
            try:
                self.server.settimeout(1.0)
                conn, addr = self.server.accept()
                logger.info(f"Enhanced connection established with {addr}")
                threading.Thread(target=self.handle_enhanced_client, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Enhanced server error: {e}")
                break

    def handle_enhanced_client(self, conn, addr):
        """Enhanced client handler with comprehensive processing"""
        buffer = b""
        conn.settimeout(10.0)
        client_start_time = time.time()
        
        try:
            while self.running:
                try:
                    data = conn.recv(self.BUFFER_SIZE)
                    if not data:
                        logger.info(f"Enhanced client {addr} disconnected")
                        break
                    
                    buffer += data
                    
                    while len(buffer) >= 4:
                        msg_length = int.from_bytes(buffer[:4], byteorder='little', signed=False)
                        
                        if len(buffer) < 4 + msg_length:
                            break
                        
                        message_bytes = buffer[4:4 + msg_length]
                        buffer = buffer[4 + msg_length:]
                        
                        try:
                            message = json.loads(message_bytes.decode('utf-8', errors='strict'))
                            response = self._process_enhanced_batch(message, addr)
                            
                            response_data = json.dumps(response, ensure_ascii=False).encode('utf-8')
                            response_header = len(response_data).to_bytes(4, byteorder='little')
                            conn.sendall(response_header + response_data)
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Enhanced JSON decode error from {addr}: {e}")
                            self._send_error_response(conn, "JSON decode error")
                        except Exception as e:
                            logger.error(f"Enhanced message processing error from {addr}: {e}")
                            self._send_error_response(conn, "Processing error")
                
                except socket.timeout:
                    continue
                except (ConnectionResetError, BrokenPipeError):
                    logger.info(f"Enhanced client {addr} connection lost")
                    break
                except Exception as e:
                    logger.error(f"Enhanced client handling error for {addr}: {e}")
                    break
                    
        finally:
            try:
                conn.close()
            except:
                pass
            duration = time.time() - client_start_time
            logger.info(f"Enhanced client {addr} handler terminated. Duration: {duration:.2f}s")

    def _process_enhanced_batch(self, batch_data: Dict[str, Any], addr: Tuple[str, int]) -> Dict[str, Any]:
        """Enhanced batch processing with aggressive exploration metrics"""
        # Update activity (only for training mode)
        if self.auto_save_manager:
            self.auto_save_manager.update_activity()
        
        # Increment request count
        self.request_count += 1
        
        logger.info(f"="*80)
        logger.info(f"[AGGRESSIVE EXPLORATION REQUEST {self.request_count}] Processing batch from {addr}")
        logger.info(f"Vehicles in batch: {len(batch_data)}")
        logger.info(f"Vehicle IDs: {list(batch_data.keys())}")
        
        batch_response = {
            "vehicles": {}, 
            "timestamp": float(time.time()),
            "request_count": int(self.request_count),
            "server_mode": "training" if self.training_mode else "production"
        }
        
        # Track batch statistics
        batch_rewards = []
        batch_states = []
        batch_actions = []
        performance_metrics = {}
        exploration_metrics = {}
        
        for vehicle_id, vehicle_data in batch_data.items():
            try:
                # Extract state
                state = [
                    float(vehicle_data.get('CBR', 0.5)),
                    float(vehicle_data.get('SNR', 15.0)),
                    float(vehicle_data.get('beaconRate', 10.0))
                ]
                
                # Get action from enhanced agent with aggressive exploration
                action = self.agent.select_action(state, self.training_mode)
                power, mcs = action
                
                # Calculate next state (simplified simulation)
                next_state = self._simulate_next_state(state, action)
                
                # Calculate enhanced reward
                reward = self.agent.calculate_enhanced_reward(state, action, next_state)
                
                logger.info(f"[AGGRESSIVE EXPLORATION VEHICLE {vehicle_id}]")
                logger.info(f"  Input State: CBR={state[0]:.3f}, SNR={state[1]:.1f}dB, BeaconRate={state[2]:.1f}Hz")
                logger.info(f"  Action: Power={power:.1f}dBm, MCS={mcs}")
                logger.info(f"  Predicted Next State: CBR={next_state[0]:.3f}, SNR={next_state[1]:.1f}dB")
                logger.info(f"  Reward: {reward:.3f}")
                
                if self.training_mode:
                    logger.info(f"  Exploration Status: Factor={self.agent.exploration_factor:.3f}, "
                               f"Epsilon={self.agent.epsilon:.3f}, PowerRand={self.agent.power_random_prob:.3f}")
                
                # Prepare response
                response_data = {
                    'transmissionPower': float(power),
                    'MCS': int(mcs),
                    'beaconRate': float(state[2]),  # Keep current beacon rate
                    'timestamp': float(vehicle_data.get('timestamp', time.time()))
                }
                
                batch_response["vehicles"][vehicle_id] = response_data
                
                # Training information (only in training mode)
                if self.training_mode:
                    # Add to replay buffer
                    done = abs(next_state[0] - sac_config.cbr_target) > 0.2  # Episode done if CBR too far from target
                    self.agent.replay_buffer.append((state, action, reward, next_state, done))
                    self.agent.cumulative_rewards.append(reward)
                    
                    # Update agent
                    update_metrics = self.agent.update()
                    if update_metrics:
                        performance_metrics = update_metrics
                    
                    # Get exploration metrics
                    exploration_metrics = {
                        'exploration_factor': self.agent.exploration_factor,
                        'epsilon': self.agent.epsilon,
                        'power_random_prob': self.agent.power_random_prob,
                        'action_noise_std': self.agent.action_noise_std,
                        'random_actions': self.agent.random_action_count,
                        'exploration_actions': self.agent.exploration_action_count,
                        'policy_actions': self.agent.policy_action_count
                    }
                    
                    # Add training information to response
                    batch_response["vehicles"][vehicle_id]['training'] = {
                        'reward': float(reward),
                        'state': state,
                        'action': [float(power), float(mcs)],
                        'next_state': next_state,
                        'done': bool(done),
                        'exploration_info': exploration_metrics
                    }
                    
                    logger.info(f"[AGGRESSIVE EXPLORATION TRAINING] Vehicle {vehicle_id}: Reward={reward:.3f}, Done={done}")
                
                # Track statistics
                batch_rewards.append(reward)
                batch_states.append(state)
                batch_actions.append(action)
                
            except Exception as e:
                error_msg = f"Error processing enhanced vehicle {vehicle_id}: {str(e)}"
                logger.error(f"[ERROR] {error_msg}")
                batch_response["vehicles"][vehicle_id] = {
                    'status': 'error',
                    'error': error_msg,
                    'timestamp': vehicle_data.get('timestamp', time.time())
                }
        
        # Enhanced batch summary with exploration info
        if batch_rewards:
            avg_reward = np.mean(batch_rewards)
            avg_cbr = np.mean([s[0] for s in batch_states])
            avg_sinr = np.mean([s[1] for s in batch_states])
            avg_power = np.mean([a[0] for a in batch_actions])
            
            logger.info(f"[AGGRESSIVE EXPLORATION BATCH SUMMARY] Request {self.request_count}:")
            logger.info(f"  Vehicles processed: {len(batch_data)}")
            logger.info(f"  Average reward: {avg_reward:.3f}")
            logger.info(f"  Average CBR: {avg_cbr:.3f}")
            logger.info(f"  Average SINR: {avg_sinr:.1f}dB")
            logger.info(f"  Average Power: {avg_power:.1f}dBm")
            
            if performance_metrics:
                logger.info(f"  Agent metrics: Alpha={performance_metrics.get('alpha', 0):.3f}")
            
            if exploration_metrics:
                logger.info(f"  Exploration metrics: Factor={exploration_metrics.get('exploration_factor', 0):.3f}, "
                           f"Epsilon={exploration_metrics.get('epsilon', 0):.3f}")
                logger.info(f"  Action counts: Random={exploration_metrics.get('random_actions', 0)}, "
                           f"Exploration={exploration_metrics.get('exploration_actions', 0)}, "
                           f"Policy={exploration_metrics.get('policy_actions', 0)}")
        
        # Add data to Excel reporter with exploration metrics
        if self.excel_reporter:
            self.excel_reporter.add_batch_data(batch_data, batch_response, performance_metrics, exploration_metrics)
        
        # Request-based periodic reporting and saving (only in training mode)
        if self.training_mode:
            if (self.request_count - self.last_report_request >= system_config.report_interval):
                self._generate_periodic_report()
                self.last_report_request = self.request_count
            
            if (self.request_count - self.last_save_request >= system_config.model_save_interval):
                self._save_models("periodic")
                self.last_save_request = self.request_count
        
        logger.info(f"[AGGRESSIVE EXPLORATION RESPONSE] Sending optimized parameters back to simulation for {len(batch_response['vehicles'])} vehicles")
        logger.info(f"="*80)
        
        return batch_response

    def _simulate_next_state(self, current_state: List[float], action: np.ndarray) -> List[float]:
        """Enhanced environment transition model"""
        try:
            cbr, sinr, beacon_rate = current_state
            power, mcs = action
            
            # Enhanced CBR simulation based on power and MCS
            power_effect = (power - 20) * 0.01  # Higher power increases CBR
            mcs_effect = (mcs - 5) * 0.005       # Higher MCS slightly increases CBR
            cbr_noise = random.uniform(-0.02, 0.02)
            new_cbr = max(0.0, min(1.0, cbr + power_effect + mcs_effect + cbr_noise))
            
            # Enhanced SINR simulation
            power_sinr_effect = (power - 20) * 0.5  # Power directly affects SINR
            sinr_noise = random.uniform(-1.0, 1.0)
            new_sinr = max(0.0, sinr + power_sinr_effect + sinr_noise)
            
            # Beacon rate remains the same for now
            new_beacon_rate = beacon_rate
            
            return [new_cbr, new_sinr, new_beacon_rate]
        except Exception as e:
            logger.error(f"Error in enhanced state simulation: {e}")
            return current_state

    def _generate_periodic_report(self):
        """Generate enhanced periodic performance reports with exploration details"""
        try:
            logger.info(f"[AGGRESSIVE EXPLORATION PERIODIC REPORT] Request {self.request_count}")
            logger.info(f"="*60)
            logger.info(f"Enhanced SAC Agent Status:")
            if self.agent.replay_buffer:
                logger.info(f"  Replay buffer size: {len(self.agent.replay_buffer)}")
            logger.info(f"  Training steps: {self.agent.training_steps}")
            logger.info(f"  Alpha value: {self.agent.log_alpha.exp().item():.4f}")
            logger.info(f"  Total episodes: {len(self.agent.cumulative_rewards)}")
            logger.info(f"  Unique actions: {len(self.agent.action_distribution)}")
            
            # Exploration status
            logger.info(f"Aggressive Exploration Status:")
            logger.info(f"  Exploration factor: {self.agent.exploration_factor:.4f}")
            logger.info(f"  Epsilon: {self.agent.epsilon:.4f}")
            logger.info(f"  Power random probability: {self.agent.power_random_prob:.4f}")
            logger.info(f"  Action noise std: {self.agent.action_noise_std:.4f}")
            
            # Action distribution
            logger.info(f"Action Selection Distribution:")
            total_actions = self.agent.random_action_count + self.agent.exploration_action_count + self.agent.policy_action_count
            if total_actions > 0:
                logger.info(f"  Random actions: {self.agent.random_action_count} ({100*self.agent.random_action_count/total_actions:.1f}%)")
                logger.info(f"  Exploration actions: {self.agent.exploration_action_count} ({100*self.agent.exploration_action_count/total_actions:.1f}%)")
                logger.info(f"  Policy actions: {self.agent.policy_action_count} ({100*self.agent.policy_action_count/total_actions:.1f}%)")
            
            # Power exploration analysis
            if self.agent.action_distribution:
                power_values = [k[0] for k in self.agent.action_distribution.keys()]
                logger.info(f"Power Exploration Analysis:")
                logger.info(f"  Min power explored: {min(power_values):.1f} dBm")
                logger.info(f"  Max power explored: {max(power_values):.1f} dBm")
                logger.info(f"  Power range covered: {max(power_values) - min(power_values):.1f} dBm")
                logger.info(f"  Power exploration percentage: {100*(max(power_values) - min(power_values))/(system_config.power_max - system_config.power_min):.1f}%")
            
            # Performance metrics
            metrics = self.agent.get_comprehensive_metrics()
            if metrics:
                logger.info(f"Enhanced Performance Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
            
            # Excel reporter status
            if self.excel_reporter:
                logger.info(f"Enhanced Excel Reporter Status:")
                logger.info(f"  Data points collected: {self.excel_reporter.vehicle_count}")
                logger.info(f"  Unique vehicles tracked: {len(self.excel_reporter.unique_vehicles)}")
                
                if self.excel_reporter.total_rewards:
                    avg_reward = np.mean(self.excel_reporter.total_rewards[-100:])
                    logger.info(f"  Recent average reward: {avg_reward:.3f}")
            
            logger.info(f"="*60)
                        
        except Exception as e:
            logger.error(f"Error generating enhanced periodic report: {e}")

    def _auto_save_callback(self, reason: str):
        """Enhanced auto-save callback with exploration tracking"""
        try:
            logger.info(f"Enhanced auto-save with exploration tracking triggered by {reason} at request {self.request_count}")
            
            # Save enhanced models
            self._save_models(reason)
            
            # Generate enhanced Excel report
            if self.excel_reporter:
                filename = self.excel_reporter.generate_comprehensive_report(f"enhanced_autosave_{reason}_req{self.request_count}")
                if filename:
                    logger.info(f"Enhanced auto-save Excel report: {filename}")
            
            # Generate performance plots
            self.agent.generate_performance_plots()
            
        except Exception as e:
            logger.error(f"Enhanced auto-save failed: {e}")

    def _save_models(self, reason: str):
        """Save enhanced models with comprehensive state and exploration parameters"""
        try:
            logger.info(f"Enhanced model saving with exploration parameters triggered by {reason}")
            
            # Save agent model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(system_config.model_save_dir, f"enhanced_sac_model_{reason}_{timestamp}.pth")
            self.agent.save_model(model_path)
            
            logger.info(f"Enhanced SAC model with exploration parameters saved:")
            logger.info(f"  Path: {model_path}")
            logger.info(f"  Training steps: {self.agent.training_steps}")
            logger.info(f"  Total episodes: {len(self.agent.cumulative_rewards)}")
            logger.info(f"  Exploration factor: {self.agent.exploration_factor:.4f}")
            logger.info(f"  Epsilon: {self.agent.epsilon:.4f}")
            logger.info(f"  Power random prob: {self.agent.power_random_prob:.4f}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced models: {e}")

    def _final_save_and_report(self):
        """Final save and enhanced report generation with exploration analysis"""
        try:
            logger.info("Generating final enhanced reports with exploration analysis and saving models...")
            
            # Save enhanced models
            self._save_models("final")
            
            # Generate final enhanced Excel report
            if self.excel_reporter:
                filename = self.excel_reporter.generate_comprehensive_report(f"final_enhanced_report_req{self.request_count}")
                if filename:
                    logger.info(f"Final enhanced Excel report: {filename}")
            
            # Generate final performance plots
            self.agent.generate_performance_plots()
            
            # Display enhanced final statistics with exploration analysis
            if self.excel_reporter:
                elapsed_time = time.time() - self.excel_reporter.start_time
                logger.info("FINAL ENHANCED SIMULATION STATISTICS WITH AGGRESSIVE EXPLORATION:")
                logger.info("="*80)
                logger.info(f"Total Requests: {self.request_count}")
                logger.info(f"Unique Vehicles: {len(self.excel_reporter.unique_vehicles)}")
                logger.info(f"Total Vehicle Interactions: {self.excel_reporter.vehicle_count}")
                logger.info(f"Duration: {elapsed_time/60:.1f} minutes")
                logger.info(f"Requests per Minute: {(self.request_count/(elapsed_time/60)):.1f}")
                
                # Enhanced statistics
                if self.excel_reporter.total_rewards:
                    logger.info(f"Average Reward: {np.mean(self.excel_reporter.total_rewards):.3f}")
                    logger.info(f"Best Reward: {np.max(self.excel_reporter.total_rewards):.3f}")
                
                if self.excel_reporter.cbr_values:
                    cbr_achievement = self.excel_reporter._calculate_cbr_achievement_rate()
                    logger.info(f"CBR Target Achievement: {cbr_achievement:.1f}%")
                
                if self.excel_reporter.action_distribution:
                    logger.info(f"Action Diversity: {len(self.excel_reporter.action_distribution)} unique actions")
                
                # Power exploration analysis
                if self.excel_reporter.power_values:
                    power_range = max(self.excel_reporter.power_values) - min(self.excel_reporter.power_values)
                    power_std = np.std(self.excel_reporter.power_values)
                    logger.info(f"Power Exploration Analysis:")
                    logger.info(f"  Power range explored: {power_range:.1f} dBm")
                    logger.info(f"  Power standard deviation: {power_std:.2f} dBm")
                    logger.info(f"  Min power used: {min(self.excel_reporter.power_values):.1f} dBm")
                    logger.info(f"  Max power used: {max(self.excel_reporter.power_values):.1f} dBm")
                    
                    # Calculate power exploration percentage
                    total_power_range = system_config.power_max - system_config.power_min
                    exploration_percentage = (power_range / total_power_range) * 100
                    logger.info(f"  Power exploration coverage: {exploration_percentage:.1f}% of total range")
                
                # Agent final exploration statistics
                total_actions = (self.agent.random_action_count + 
                               self.agent.exploration_action_count + 
                               self.agent.policy_action_count)
                
                logger.info(f"SAC Agent Final Status:")
                logger.info(f"  Training Steps: {self.agent.training_steps:,}")
                logger.info(f"  Final Alpha: {self.agent.log_alpha.exp().item():.4f}")
                logger.info(f"  Final Exploration Factor: {self.agent.exploration_factor:.4f}")
                logger.info(f"  Final Epsilon: {self.agent.epsilon:.4f}")
                logger.info(f"  Final Power Random Probability: {self.agent.power_random_prob:.4f}")
                logger.info(f"  Final Action Noise Std: {self.agent.action_noise_std:.4f}")
                
                if total_actions > 0:
                    logger.info(f"Action Selection Distribution:")
                    logger.info(f"  Random actions: {self.agent.random_action_count:,} ({100*self.agent.random_action_count/total_actions:.1f}%)")
                    logger.info(f"  Exploration actions: {self.agent.exploration_action_count:,} ({100*self.agent.exploration_action_count/total_actions:.1f}%)")
                    logger.info(f"  Policy actions: {self.agent.policy_action_count:,} ({100*self.agent.policy_action_count/total_actions:.1f}%)")
                
                logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Error in enhanced final save and report: {e}")

    def _send_error_response(self, conn: socket.socket, error_msg: str):
        """Send error response to client"""
        try:
            error_response = json.dumps({
                'status': 'error',
                'error': error_msg,
                'timestamp': time.time()
            }).encode('utf-8')
            
            header = len(error_response).to_bytes(4, byteorder='little')
            conn.sendall(header + error_response)
        except:
            pass

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Enhanced server received signal {signum}, initiating graceful shutdown...")
        self.running = False
        self._shutdown()

    def _shutdown(self):
        """Enhanced graceful shutdown"""
        logger.info("Initiating enhanced server shutdown...")
        self.running = False
        
        try:
            # Stop auto-save manager (only if in training mode)
            if self.auto_save_manager:
                self.auto_save_manager.stop()
            
            # Final save and report (only if in training mode)
            if self.training_mode:
                self._final_save_and_report()
            
            # Cleanup agent
            self.agent.cleanup()
            
            # Close server socket
            try:
                self.server.close()
            except:
                pass
            
            logger.info("Enhanced server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during enhanced shutdown: {e}")

    def stop(self):
        """External stop method"""
        self._shutdown()

# Global reference for cleanup
enhanced_rl_server = None

def save_model_on_exit():
    """Save model on exit - only in training mode"""
    if enhanced_rl_server and enhanced_rl_server.training_mode:
        logger.info("Saving enhanced model before exiting...")
        enhanced_rl_server.agent.save_model()

atexit.register(save_model_on_exit)

# ==============================
# Main Execution with Enhanced Arguments
# ==============================

def parse_enhanced_arguments():
    """Parse enhanced command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Conventional SAC with Aggressive Power Exploration')
    
    # Server configuration
    parser.add_argument('--host', default=system_config.host, 
                       help=f'Server host (default: {system_config.host})')
    parser.add_argument('--port', type=int, default=system_config.port, 
                       help=f'Server port (default: {system_config.port})')
    
    # SAC configuration
    parser.add_argument('--cbr-target', type=float, default=sac_config.cbr_target,
                       help=f'CBR target (default: {sac_config.cbr_target})')
    parser.add_argument('--learning-rate', type=float, default=sac_config.lr_actor,
                       help=f'Learning rate (default: {sac_config.lr_actor})')
    parser.add_argument('--hidden-units', type=int, default=sac_config.hidden_units,
                       help=f'Hidden units (default: {sac_config.hidden_units})')
    
    # Aggressive exploration configuration
    parser.add_argument('--initial-exploration', type=float, default=sac_config.initial_exploration_factor,
                       help=f'Initial exploration factor (default: {sac_config.initial_exploration_factor})')
    parser.add_argument('--initial-epsilon', type=float, default=sac_config.initial_epsilon,
                       help=f'Initial epsilon for random actions (default: {sac_config.initial_epsilon})')
    parser.add_argument('--power-random-prob', type=float, default=sac_config.power_random_prob,
                       help=f'Power random probability (default: {sac_config.power_random_prob})')
    parser.add_argument('--exploration-noise', type=float, default=sac_config.exploration_noise_scale,
                       help=f'Exploration noise scale (default: {sac_config.exploration_noise_scale})')
    
    # Auto-save configuration (only relevant for training mode)
    parser.add_argument('--timeout', type=int, default=system_config.auto_save_timeout,
                       help=f'Auto-save timeout in minutes (default: {system_config.auto_save_timeout})')
    
    # Feature toggles
    parser.add_argument('--disable-tensorboard', action='store_true',
                       help='Disable TensorBoard logging')
    parser.add_argument('--disable-excel', action='store_true',
                       help='Disable Excel reporting')
    
    # Debugging
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility (default: 0)')
    
    return parser.parse_args()

def setup_enhanced_configuration(args):
    """Setup enhanced global configuration with aggressive exploration"""
    global ENABLE_TENSORBOARD, ENABLE_EXCEL_REPORTING
    
    # Operation mode is set at the top of the script
    training_mode = (OPERATION_MODE == "training")
    
    if training_mode:
        logger.info("Running in TRAINING mode with AGGRESSIVE EXPLORATION")
    else:
        logger.info("Running in PRODUCTION mode (inference only)")
    
    # Handle feature toggles
    if args.disable_tensorboard:
        ENABLE_TENSORBOARD = False
        logger.info("TensorBoard logging disabled")
    
    if args.disable_excel:
        ENABLE_EXCEL_REPORTING = False
        logger.info("Excel reporting disabled")
    
    # Debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Update configurations
    sac_config.cbr_target = args.cbr_target
    sac_config.lr_actor = args.learning_rate
    sac_config.lr_critic = args.learning_rate
    sac_config.hidden_units = args.hidden_units
    
    # Update aggressive exploration parameters
    sac_config.initial_exploration_factor = args.initial_exploration
    sac_config.initial_epsilon = args.initial_epsilon
    sac_config.power_random_prob = args.power_random_prob
    sac_config.exploration_noise_scale = args.exploration_noise
    
    system_config.host = args.host
    system_config.port = args.port
    system_config.auto_save_timeout = args.timeout
    
    return training_mode

def print_enhanced_startup_banner(args, training_mode):
    """Print enhanced startup configuration banner with aggressive exploration details"""
    logger.info("=" * 80)
    logger.info("ENHANCED CONVENTIONAL SAC WITH AGGRESSIVE POWER EXPLORATION")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Server: {args.host}:{args.port}")
    logger.info(f"  Mode: {'TRAINING' if training_mode else 'PRODUCTION'}")
    if training_mode:
        logger.info(f"  Auto-save timeout: {args.timeout} minutes")
    
    logger.info(f"Enhanced SAC Parameters:")
    logger.info(f"  CBR target: {sac_config.cbr_target}")
    logger.info(f"  Learning rate: {sac_config.lr_actor}")
    logger.info(f"  Hidden units: {sac_config.hidden_units}")
    logger.info(f"  Buffer size: {sac_config.buffer_size:,}")
    logger.info(f"  Batch size: {sac_config.batch_size}")
    
    logger.info(f"AGGRESSIVE EXPLORATION Parameters:")
    logger.info(f"  Initial exploration factor: {sac_config.initial_exploration_factor}")
    logger.info(f"  Initial epsilon (random actions): {sac_config.initial_epsilon}")
    logger.info(f"  Power random probability: {sac_config.power_random_prob}")
    logger.info(f"  Exploration noise scale: {sac_config.exploration_noise_scale}")
    logger.info(f"  Power exploration bonus: {sac_config.power_exploration_bonus}")
    logger.info(f"  Action noise std: {sac_config.action_noise_std}")
    logger.info(f"  Exploration decay: {sac_config.exploration_decay}")
    logger.info(f"  Epsilon decay: {sac_config.epsilon_decay}")
    logger.info(f"  Min exploration: {sac_config.min_exploration}")
    logger.info(f"  Min epsilon: {sac_config.min_epsilon}")
    
    logger.info(f"Features:")
    logger.info(f"  TensorBoard: {'Enabled' if ENABLE_TENSORBOARD else 'Disabled'}")
    logger.info(f"  Excel reporting: {'Enabled' if ENABLE_EXCEL_REPORTING else 'Disabled'}")
    logger.info(f"  Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    
    logger.info(f"Aggressive Exploration Features:")
    logger.info(f"  - Multi-strategy exploration: Random, Power-specific, Policy-based")
    logger.info(f"  - Epsilon-greedy for complete randomization")
    logger.info(f"  - Power-specific aggressive exploration")
    logger.info(f"  - Multiple layers of exploration noise")
    logger.info(f"  - Comprehensive exploration tracking and analysis")
    logger.info(f"  - Fast exploration decay for short training periods")
    logger.info(f"  - Enhanced action distribution analysis")
    logger.info(f"  - Power exploration coverage metrics")
    
    logger.info(f"Enhanced Reporting Features:")
    logger.info(f"  - Comprehensive Excel reports with exploration analysis")
    logger.info(f"  - Real-time performance plots with exploration metrics")
    logger.info(f"  - TensorBoard integration for training and exploration metrics")
    logger.info(f"  - Power distribution entropy calculation")
    logger.info(f"  - Action selection strategy tracking")
    logger.info(f"  - Exploration coverage percentage analysis")
    
    if not training_mode:
        logger.info(f"Production Mode Features:")
        logger.info(f"  - Automatic loading of latest trained models")
        logger.info(f"  - Deterministic action selection for consistency")
        logger.info(f"  - No model updates or exploration")
        logger.info(f"  - Optimized inference performance")
    
    logger.info("=" * 80)
    logger.info("POWER EXPLORATION OPTIMIZATION:")
    logger.info(f"  Power range: {system_config.power_min}-{system_config.power_max} dBm")
    logger.info(f"  Target: Aggressive exploration across full power range")
    logger.info(f"  Strategy: Multi-layered exploration with power-specific bonuses")
    logger.info(f"  Expected: Fast convergence with comprehensive power coverage")
    logger.info("=" * 80)

if __name__ == "__main__":
    try:
        # Parse and validate enhanced arguments
        args = parse_enhanced_arguments()
        
        # Validate enhanced arguments
        if not (1024 <= args.port <= 65535):
            logger.error("Port must be between 1024 and 65535")
            sys.exit(1)
        
        if args.timeout <= 0:
            logger.error("Timeout must be positive")
            sys.exit(1)
        
        # Validate exploration parameters
        if not (0.0 <= args.initial_epsilon <= 1.0):
            logger.error("Initial epsilon must be between 0.0 and 1.0")
            sys.exit(1)
        
        if not (0.0 <= args.power_random_prob <= 1.0):
            logger.error("Power random probability must be between 0.0 and 1.0")
            sys.exit(1)
        
        if args.initial_exploration < 1.0:
            logger.error("Initial exploration factor should be >= 1.0 for effective exploration")
            sys.exit(1)
        
        # Setup enhanced configuration
        training_mode = setup_enhanced_configuration(args)
        
        # Set random seed
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")
        
        # Print enhanced startup banner
        print_enhanced_startup_banner(args, training_mode)
        
        # Create and start enhanced server
        enhanced_rl_server = EnhancedRLServer(
            host=args.host,
            port=args.port,
            training_mode=training_mode,
            timeout_minutes=args.timeout
        )
        
        logger.info("Starting Enhanced Conventional SAC server with AGGRESSIVE POWER EXPLORATION...")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("Server optimized for fast power exploration and comprehensive performance reporting")
        
        if training_mode:
            logger.info("AGGRESSIVE EXPLORATION MODE ACTIVE:")
            logger.info(f"  - {sac_config.initial_epsilon*100:.0f}% initial random actions")
            logger.info(f"  - {sac_config.power_random_prob*100:.0f}% power-specific randomization")
            logger.info(f"  - {sac_config.initial_exploration_factor}x exploration amplification")
            logger.info(f"  - Multi-layer exploration noise")
            logger.info("  - Expected fast convergence with comprehensive power coverage")
        
        # Start enhanced server
        enhanced_rl_server.start()
        
    except KeyboardInterrupt:
        logger.info("Enhanced server interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("Enhanced server shutdown complete")