import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_results.csv')
df['ma10'] = df['reward'].rolling(window=100, min_periods=1).mean()

plt.figure()
plt.plot(df['episode'], df['reward'], color='C0')
plt.plot(df['episode'], df['ma10'], linewidth=2, color='C1')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DDPG Training Progress Q2')

plt.savefig('training_results.png')
print("Plot saved as training_results.png")
