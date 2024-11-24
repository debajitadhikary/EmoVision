import matplotlib.pyplot as plt

# Example latency data (milliseconds)
latency_data = [16.2, 16.1, 16.4, 15.9, 16.0, 16.3, 15.8, 16.2, 16.5, 16.1]

# Plot latency vs frame number
plt.plot(latency_data)
plt.title('Latency (ms) per Frame')
plt.xlabel('Frame Number')
plt.ylabel('Latency (ms)')
plt.grid(True)
plt.show()
