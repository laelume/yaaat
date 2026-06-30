## BASIC SKETCH - SHOWING INTERACTIVE CLUSTERING AND COMPANION SPECTROGRAM INSPECTION TOOL

# Use TkAgg backend
%matplotlib tk

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pysoniq
import pysoniq.display

# Reduce dimensions
print("Reducing dimensions...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors = ['red', 'blue', 'green', 'orange', 'purple']

for cluster_id in range(n_clusters):
    mask = clusters == cluster_id
    ax1.scatter(
        embeddings_2d[mask, 0], 
        embeddings_2d[mask, 1],
        c=colors[cluster_id],
        label=f'Cluster {cluster_id} ({np.sum(mask)} files)',
        alpha=0.6,
        s=50
    )

ax1.set_xlabel('t-SNE dimension 1')
ax1.set_ylabel('t-SNE dimension 2')
ax1.set_title('Click on a point to see spectrogram')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Show initial spectrogram
y, sr = pysoniq.load_audio(filenames[0])

S = pysoniq.fourier.magnitude_stft(y, n_fft=512, hop_length=64)
D = pysoniq.fourier.amplitude_to_db(S, ref=np.max(S))

img = pysoniq.visualize.spectroviz(D, sr=sr, hop_length=64, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
ax2.set_title(f'Click a point | {filenames[0].split(chr(92))[-1]}')

# Create colorbar once and store reference
cbar = plt.colorbar(img, ax=ax2, format='%+2.0f dB')

# Click handler
def onclick(event):
    global cbar
    
    if event.inaxes != ax1:
        return
    
    # Find closest point
    click_x, click_y = event.xdata, event.ydata
    if click_x is None or click_y is None:
        return
        
    distances = np.sqrt((embeddings_2d[:, 0] - click_x)**2 + (embeddings_2d[:, 1] - click_y)**2)
    closest_idx = np.argmin(distances)
    
    # Only trigger if click is reasonably close
    if distances[closest_idx] < 1.0:
        audio_file = filenames[closest_idx]
        cluster_id = clusters[closest_idx]
        
        print(f"Loading: {audio_file.split(chr(92))[-1]}")
        
        # Load audio
        y, sr = pysoniq.load_audio(audio_file)
        
        S = pysoniq.fourier.magnitude_stft(y, n_fft=512, hop_length=64)
        D = pysoniq.fourier.amplitude_to_db(S, ref=np.max(S))
        
        # Remove old colorbar
        cbar.remove()
        
        # Clear and redraw spectrogram
        ax2.clear()
        img = pysoniq.visualize.spectroviz(D, sr=sr, hop_length=64, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
        ax2.set_title(f'Cluster {cluster_id}: {audio_file.split(chr(92))[-1]}')
        
        # Create new colorbar
        cbar = plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        fig.canvas.draw()
        fig.canvas.flush_events()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()