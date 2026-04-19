import pandas as pd
import matplotlib.pyplot as plt

# Replace with your G4BL trace file
filename = "Ev3Trk1.txt"

# Read whitespace-separated file
df = pd.read_csv(filename, delim_whitespace=True, comment='#', header=None)

# Assign column names if you know the structure
df.columns = ['#','x', 'y', 'z', 'Px', 'Py', 'Pz', 't' 'PDGid', 'EventID', 'TrackID', 'ParentID', 'Weight', 'Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez']

# Choose a specific particle
track_id = 1  # for example
particle_df = df[df['TrackID'] == track_id]

# Plot x vs z
plt.plot(particle_df['z'], particle_df['x'], marker='o', linestyle='-')
plt.xlabel('z (mm)')
plt.ylabel('x (mm)')
plt.title(f'Track {track_id} in centerline coordinates')
plt.grid(True)
plt.show()