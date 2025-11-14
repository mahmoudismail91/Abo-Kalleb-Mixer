import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
import os
import random
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import wave
import collections
import math
import pedalboard as pb
import warnings
import re  # For smart filenames
# from scipy.signal import resample  # --- REMOVED: Too CPU-heavy ---
from concurrent.futures import ThreadPoolExecutor
import subprocess
import shutil  # For export

# --- VJ MODE IMPORTS ---
try:
    import cv2  # For OpenCV (pip install opencv-python)
    from PIL import Image, ImageTk, ImageSequence  # For Pillow (pip install pillow)
    VJ_LIBS_INSTALLED = True
except ImportError:
    print("WARNING: VJ libraries not found. Run 'pip install opencv-python pillow' to enable visuals.")
    VJ_LIBS_INSTALLED = False
# --- END NEW IMPORTS ---


# Suppress a harmless warning from pedalboard
warnings.filterwarnings("ignore", category=UserWarning, module='pedalboard')

# --- TRACK CLASS ---
class Track:
    def __init__(self, audio_segment, name, target_rate=44100):
        self.name = name
        self.original_audio = audio_segment.set_frame_rate(target_rate).set_channels(2).set_sample_width(2)
        
        int_samples = np.frombuffer(self.original_audio.raw_data, dtype=np.int16).reshape(-1, 2)
        # --- Store samples as float32 for processing ---
        self.samples = int_samples.astype(np.float32) / 32768.0 
        
        self.length = len(self.samples)
        self.pointer = 0
        self.is_active = True
        
        self.is_waiting_for_sync = False
        
        self.base_volume = tk.DoubleVar(value=0.7)
        self.base_pan = tk.DoubleVar(value=0.0)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.progress_var = tk.DoubleVar(value=0)

    # --- NEW: Fast Linear Interpolation Resampler ---
    def _resample_linear(self, in_chunk, target_len):
        """
        A fast linear interpolation resampler.
        in_chunk: The numpy array (N, 2) of audio data to resample.
        target_len: The desired output length (chunk_size).
        """
        in_len = len(in_chunk)
        if in_len == 0:
            return np.zeros((target_len, 2), dtype=np.float32)
            
        # Create an array of floating-point indices for sampling
        indices = np.linspace(0, in_len - 1, target_len)
        
        # Find the integer indices and fractional parts
        idx_floor = indices.astype(np.int32)
        idx_ceil = np.minimum(idx_floor + 1, in_len - 1)
        frac = (indices - idx_floor).reshape(-1, 1) # Reshape for broadcasting
        
        # Perform linear interpolation
        # (sample_at_ceil * frac) + (sample_at_floor * (1.0 - frac))
        out_chunk = (in_chunk[idx_ceil] * frac) + (in_chunk[idx_floor] * (1.0 - frac))
        
        return out_chunk.astype(np.float32)
    # --- END NEW RESAMPLER ---

    def get_chunk(self, chunk_size, params, mode):
        if not self.is_active:
            return None
            
        if mode == 'sync' and self.is_waiting_for_sync:
            return np.zeros((chunk_size, 2), dtype=np.float32) 
        
        if params is None:
            return np.zeros((chunk_size, 2), dtype=np.float32)
            
        speed, volume, pan = params
        
        read_size = int(chunk_size * speed)
        chunk_end = self.pointer + read_size
        
        if chunk_end >= self.length:
            if mode == 'sync':
                chunk = self.samples[self.pointer : self.length]
                self.is_waiting_for_sync = True 
                self.pointer = 0
                padding_size = read_size - len(chunk)
                if padding_size > 0:
                    padding = np.zeros((padding_size, 2), dtype=np.float32) 
                    chunk = np.concatenate([chunk, padding])
            else:
                chunk = self.samples[self.pointer:self.length]
                remaining_after_loop = read_size - len(chunk)
                if remaining_after_loop > 0:
                    loop_chunk = self.samples[:remaining_after_loop]
                    chunk = np.concatenate([chunk, loop_chunk])
                self.pointer = remaining_after_loop
        else:
            chunk = self.samples[self.pointer : chunk_end]
            self.pointer = chunk_end

        # --- OPTIMIZED: Use new fast resampler instead of SciPy ---
        if speed != 1.0:
            if len(chunk) > 1:
                # Use fast linear interpolation
                chunk = self._resample_linear(chunk, chunk_size)
            elif len(chunk) == 1:
                chunk = np.tile(chunk, (chunk_size, 1))
            else:
                return np.zeros((chunk_size, 2), dtype=np.float32) 
        
        if len(chunk) < chunk_size:
            padding = np.zeros((chunk_size - len(chunk), 2), dtype=np.float32) 
            chunk = np.concatenate([chunk, padding])
        elif len(chunk) > chunk_size:
            chunk = chunk[:chunk_size]

        # --- OPTIMIZED: Pan and volume logic ---
        # Pre-calculate gains
        if pan == 0.0:
            gain_left = volume
            gain_right = volume
        else:
            # Use constant power panning
            pan_rad = (pan * 0.5 + 0.5) * (math.pi / 2.0)
            gain_left = math.cos(pan_rad) * volume
            gain_right = math.sin(pan_rad) * volume
            
        # Apply gains (vectorized)
        chunk[:, 0] *= gain_left
        chunk[:, 1] *= gain_right

        # Clipping is still necessary after gain
        chunk = np.clip(chunk, -1.0, 1.0) 
        
        return chunk

# --- MIXER ---
class AlHutMixer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🐋 AL HUT الحوت Abo Kalleb Mixer أبو كلب 🐋")
        self.geometry("900x700")
        self.minsize(800, 600)
        
        # Audio settings
        self.RATE = 44100
        self.CHUNK = 2048
        self.CHANNELS = 2
        
        self.audio_active = threading.Event()
        self.stream = None
        
        self.data_lock = threading.Lock()
        
        self.cache_thread = None
        self.digger_thread = None
        
        # Audio Pre-load Buffer
        self.is_currently_swapping = False
        self.next_track_buffer = None 
        self.preload_event = threading.Event()
        
        # --- NEW: Visuals Data ---
        self.all_visual_files = []
        self.visual_thread = None
        self.visual_stop_event = threading.Event()
        self.current_visual_path = None
        self.visual_label_width = 640  # Default
        self.visual_label_height = 360  # Default
        # --- END NEW ---
        
        # --- NEW: Export Data (from mixer69) ---
        self.exporting = False
        self.export_video_frames = []
        self.export_audio_frames = []
        self.export_start_time = None
        self.export_timer = None
        self.export_thread = None
        self.export_output_filename = None
        self.current_frame_lock = threading.Lock()
        self.current_pil_image = None # For sharing frame to export thread
        # --- END NEW EXPORT ---
        
        # Data
        self.all_audio_files = []
        self.active_tracks = [] 
        self.track_params = {} 
        self.track_frames = {}
        
        self.slider_cache = {}
        
        self.recording_event = threading.Event()
        self.record_frames = collections.deque(maxlen=5000)
        self.record_filename = None
        
        # UI
        self.master_volume = tk.DoubleVar(value=0.7)
        self.reverb_var = tk.DoubleVar(value=0.0) 
        self.delay_var = tk.DoubleVar(value=0.0)
        self.drive_var = tk.DoubleVar(value=0.0)
        self.bpm_var = tk.DoubleVar(value=120.0)
        
        # Ambient Engine UI
        self.is_ambient_mode = tk.BooleanVar(value=False)
        self.ambient_density_var = tk.DoubleVar(value=20.0)
        
        # Thread-safe caches
        self.master_volume_cache = self.master_volume.get()
        self.reverb_cache = self.reverb_var.get()
        self.delay_cache = self.delay_var.get()
        self.drive_cache = self.drive_var.get()
        self.bpm_cache = self.bpm_var.get()
        self.is_ambient_mode_cache = self.is_ambient_mode.get()
        
        # Master Clock
        self.master_sample_counter = 0
        self.samples_per_bar = self.calculate_samples_per_bar(self.bpm_cache)
        
        # Master Effects
        self.reverb = pb.Reverb(room_size=0.6, wet_level=0.0)
        self.delay = pb.Delay(delay_seconds=0.5, feedback=0.4, mix=0.0)
        self.distortion = pb.Distortion(drive_db=0.0)
        
        self.master_effects = pb.Pedalboard([
            self.distortion,
            self.reverb,
            self.delay
        ])
        
        # Dirty flags for effects
        self.effects_dirty = True
        
        # Event for param updates
        self.param_update_event = threading.Event()
        
        self.create_layout()
        
        # Trace variables to set param_update_event
        self.master_volume.trace('w', self._on_param_change)
        self.reverb_var.trace('w', self._on_param_change)
        self.delay_var.trace('w', self._on_param_change)
        self.drive_var.trace('w', self._on_param_change)
        self.bpm_var.trace('w', self._on_param_change)
        self.is_ambient_mode.trace('w', self._on_param_change)
        self.ambient_density_var.trace('w', self._on_param_change)
        
        self.title_colors = ['#007FFF', '#009FFF', '#00BFFF', '#00FFFF', '#00BFFF', '#009FFF']
        self.title_color_index = 0
        self.animate_title()

        # Bind resize
        self.visual_frame.bind("<Configure>", self._update_visual_size)

    def _update_visual_size(self, event=None):
        # Add a small buffer to prevent feedback loop
        if abs(self.visual_label_width - event.width) > 2 or abs(self.visual_label_height - event.height) > 2:
            self.visual_label_width = event.width
            self.visual_label_height = event.height
    
    def animate_title(self):
        try:
            color = self.title_colors[self.title_color_index]
            self.title_label.config(fg=color)
            self.title_color_index = (self.title_color_index + 1) % len(self.title_colors)
            delay = 200 if self.is_ambient_mode.get() else 500  # Faster in ambient
            self.after(delay, self.animate_title)
        except tk.TclError:
            pass # App closing
    
    def _on_param_change(self, *args):
        self.param_update_event.set()
    
    def calculate_samples_per_bar(self, bpm):
        if bpm <= 0: bpm = 1
        seconds_per_beat = 60.0 / bpm
        seconds_per_bar = seconds_per_beat * 4.0
        return int(seconds_per_bar * self.RATE)

    def create_layout(self):
        self.configure(bg='lightblue') # Set main window background
        
        # Header
        self.title_label = tk.Label(self, text="🐋 AL HUT الحوت Abo Kalleb Mixer أبو كلب 🐋", font=("Arial", 20, "bold"), 
                                    fg='blue', bg='lightblue')
        self.title_label.pack(pady=(10, 5))
        
        # --- Main Paned Window (splits visuals from controls) ---
        self.main_pane = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED, bg='lightblue', sashwidth=6)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # --- Top Pane: Visuals ---
        self.visual_frame = tk.Frame(self.main_pane, bg='black', relief=tk.SUNKEN, borderwidth=2)
        self.visual_label = tk.Label(self.visual_frame, bg='black')
        self.visual_label.pack(fill=tk.BOTH, expand=True)
        self.main_pane.add(self.visual_frame, height=250, minsize=100)

        # --- Bottom Pane: All Controls ---
        self.controls_frame = tk.Frame(self.main_pane, bg='lightblue')
        self.main_pane.add(self.controls_frame, minsize=400)
        
        # Controls
        control_frame = tk.Frame(self.controls_frame, bg='lightblue')
        control_frame.pack(fill=tk.X, pady=5)
        
        # Left
        left_ctrl = tk.Frame(control_frame, bg='lightblue')
        left_ctrl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        tk.Button(left_ctrl, text="📁 Load Music", command=self.load_audio_folder, 
                 font=('Arial', 11)).pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(left_ctrl, text="🎬 Load Visuals", command=self.load_visuals_folder, 
                 font=('Arial', 11), bg='#FFD700').pack(fill=tk.X, padx=5, pady=5)
                 
        tk.Button(left_ctrl, text="🎲 Randomize Volumes", command=self.randomize_volumes, 
                 font=('Arial', 10)).pack(fill=tk.X, padx=5, pady=2)
        
        # Right
        right_ctrl = tk.Frame(control_frame, bg='lightblue')
        right_ctrl.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        self.right_ctrl_frame = right_ctrl # --- Save reference ---
        
        mix_btn_frame = tk.Frame(right_ctrl)
        mix_btn_frame.pack(fill=tk.X, padx=5)
        self.play_button = tk.Button(mix_btn_frame, text="▶ PLAY", command=self.toggle_mix, 
                                   font=('Arial', 12, 'bold'), bg='green', fg='white')
        self.play_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        
        tk.Button(mix_btn_frame, text="⏭ NEXT", command=self.start_new_mix, 
                 font=('Arial', 10), bg='orange', fg='white').pack(side=tk.RIGHT)
        
        tk.Button(right_ctrl, text="🔄 RESET Speeds/Pans", command=self.reset_speeds_pans, 
                 font=('Arial', 10), bg='gray', fg='white').pack(fill=tk.X, padx=5, pady=2)

        self.record_button = tk.Button(right_ctrl, text="🔴 RECORD Audio", command=self.toggle_record, 
                                     font=('Arial', 10, 'bold'), bg='darkred', fg='white')
        self.record_button.pack(fill=tk.X, padx=5, pady=5)

        # --- MODIFIED: Calls new export function ---
        self.export_button = tk.Button(right_ctrl, text="📹 Export A/V", command=self.toggle_export, 
                                       font=('Arial', 10, 'bold'), bg='purple', fg='white')
        self.export_button.pack(fill=tk.X, padx=5, pady=2)
        
        self.export_time_label = tk.Label(right_ctrl, text="", font=('Arial', 9, 'bold'), bg='lightblue', fg='purple')
        self.export_time_label.pack(fill=tk.X, padx=5)
        # --- END MODIFIED ---
        
        # --- Master Engine Control ---
        engine_frame = tk.Frame(self.controls_frame, bg='lightgray', relief=tk.RIDGE, borderwidth=2)
        engine_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(engine_frame, text="--- MASTER ENGINE ---", font=("Arial", 10, 'bold'), bg='lightgray').pack()
        
        self.mode_button = tk.Button(engine_frame, text="Mode: SYNC", font=('Arial', 12, 'bold'), 
                                     bg='cyan', command=self.toggle_mode)
        self.mode_button.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        
        self.bpm_slider = tk.Scale(engine_frame, from_=60, to=180, resolution=0.5, orient=tk.HORIZONTAL, 
                                  variable=self.bpm_var, label="Global BPM", length=200)
        self.bpm_slider.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.density_slider = tk.Scale(engine_frame, from_=20, to=60, resolution=1, orient=tk.HORIZONTAL,
                                       variable=self.ambient_density_var, label="Ambient Density (sec)", length=200)
        self.density_slider.pack(side=tk.LEFT, padx=5, pady=5)
        
        # --- Master Effects ---
        effects_frame = tk.Frame(self.controls_frame, bg='lightblue')
        effects_frame.pack(fill=tk.X, pady=5)

        tk.Scale(effects_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, 
                variable=self.drive_var, label="Drive", length=100).pack(side=tk.RIGHT, padx=5)
        
        tk.Scale(effects_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, 
                variable=self.delay_var, label="Delay", length=150).pack(side=tk.RIGHT, padx=5)
        
        tk.Scale(effects_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, 
                variable=self.reverb_var, label="Reverb", length=150).pack(side=tk.RIGHT, padx=5)
        
        tk.Scale(effects_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, 
                variable=self.master_volume, label="Master Volume", length=200).pack(side=tk.RIGHT)
        
        self.status_label = tk.Label(effects_frame, text="Ready", font=("Arial", 11, 'bold'), bg='lightblue')
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Tracks display
        track_container = tk.Frame(self.controls_frame, bg='lightblue')
        track_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.track_canvas = tk.Canvas(track_container, bg='lightblue', highlightthickness=0)
        self.track_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(track_container, orient=tk.VERTICAL, command=self.track_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.track_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.tracks_frame = tk.Frame(self.track_canvas, bg='lightblue')
        self.track_canvas.create_window((0,0), window=self.tracks_frame, anchor="nw")
        self.tracks_frame.bind("<Configure>", lambda e: self.track_canvas.configure(scrollregion=self.track_canvas.bbox("all")))
        
        self.update_ui_mode()
    
    
    # --- FILE LOADING ---
    def load_audio_folder(self):
        folder = filedialog.askdirectory(title="Select Music Folder")
        if folder:
            self.status_label.config(text="Scanning audio...")
            self.scan_progress = ttk.Progressbar(self.controls_frame, mode='indeterminate', length=300)
            self.scan_progress.pack(pady=5)
            self.scan_progress.start()
            
            def scan_and_stop():
                self._scan_audio(folder)
                self.after(0, lambda: self.scan_progress.stop())
                self.after(0, lambda: self.scan_progress.destroy())
            
            threading.Thread(target=scan_and_stop, daemon=True).start()
    
    def _scan_audio(self, folder):
        exts = ('.mp3','.wav','.flac','.ogg','.m4a')
        files = []
        max_size_bytes = 500 * 1024 * 1024  # 500MB
        
        for root, _, fs in os.walk(folder):
            for f in fs:
                if f.lower().endswith(exts):
                    fp = os.path.join(root, f)
                    try:
                        if os.path.getsize(fp) < max_size_bytes:
                            files.append(fp)
                        else:
                            print(f"Skipping large file: {f} (>500MB)")
                    except OSError as e:
                        print(f"Error accessing file {fp}: {e}")
                        pass
                        
        self.all_audio_files = files
        self.after(0, lambda: self.status_label.config(text=f"Found {len(self.all_audio_files)} audio tracks"))
    
    def load_visuals_folder(self):
        if not VJ_LIBS_INSTALLED:
            messagebox.showerror("Error", "Visual libraries not found.\nPlease run: pip install opencv-python pillow")
            return
            
        folder = filedialog.askdirectory(title="Select Visuals Folder")
        if folder:
            self.status_label.config(text="Scanning visuals...")
            self.visual_scan_progress = ttk.Progressbar(self.controls_frame, mode='indeterminate', length=300)
            self.visual_scan_progress.pack(pady=5)
            self.visual_scan_progress.start()
            
            def scan_and_stop():
                self._scan_visuals(folder)
                self.after(0, lambda: self.visual_scan_progress.stop())
                self.after(0, lambda: self.visual_scan_progress.destroy())
            
            threading.Thread(target=scan_and_stop, daemon=True).start()

    def _scan_visuals(self, folder):
        exts = ('.mp4', '.mkv', '.avi', '.webm', '.gif', '.jpg', '.jpeg', '.png')
        files = []
        for root, _, fs in os.walk(folder):
            for f in fs:
                if f.lower().endswith(exts):
                    fp = os.path.join(root, f)
                    files.append(fp)
                        
        self.all_visual_files = files
        self.after(0, lambda: self.status_label.config(text=f"Found {len(self.all_visual_files)} visuals"))
    
    # --- MIXING ---
    def toggle_mix(self):
        if not self.all_audio_files:
            messagebox.showerror("Error", "No music files loaded!")
            return
        
        if not self.audio_active.is_set():
            self.start_mix()
        else:
            self.stop_mix()
    
    def start_mix(self):
        self.audio_active.set()
        self.play_button.config(text="⏸ STOP", bg='red')
        
        self.master_sample_counter = 0
        self.is_currently_swapping = False
        
        self.next_track_buffer = None
        self.preload_event.clear()
        
        self.cache_thread = threading.Thread(target=self.cache_params_thread, daemon=True)
        self.digger_thread = threading.Thread(target=self.ambient_digger_thread, daemon=True)
        self.cache_thread.start()
        self.digger_thread.start()
        
        self.update_slider_cache()
        
        self.load_initial_tracks()
        
        self.load_next_visual()
        
        self.preload_event.set()
    
    def load_initial_tracks(self):
        try:
            track_count = 4 if self.is_ambient_mode.get() else 3
            track_count = min(track_count, len(self.all_audio_files))
        except:
            track_count = 0
        
        if track_count == 0:
            self.status_label.config(text="No tracks loaded to play")
            if self.audio_active.is_set(): self.stop_mix()
            return

        selected = random.sample(self.all_audio_files, track_count)
        self.status_label.config(text=f"Loading {track_count} tracks...")
        
        threading.Thread(target=self._load_and_start, args=(selected,), daemon=True).start()

    def _load_single_track(self, file_path):
        try:
            max_duration_sec = 600  # 10 minutes
            audio = None
            
            if file_path.lower().endswith('.wav'):
                with wave.open(file_path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
                if duration > max_duration_sec:
                    print(f"Trimming long WAV: {os.path.basename(file_path)} ({duration:.0f}s -> {max_duration_sec}s)")
                    audio = AudioSegment.from_file(file_path)[:max_duration_sec * 1000]
                else:
                    audio = AudioSegment.from_file(file_path)
            else:
                audio = AudioSegment.from_file(file_path)
                if len(audio) / 1000 > max_duration_sec:
                    print(f"Trimming long track: {os.path.basename(file_path)}")
                    audio = audio[:max_duration_sec * 1000]
            
            track = Track(audio, os.path.basename(file_path)[:30], self.RATE)
            return track
        except MemoryError:
            print(f"MemoryError loading {os.path.basename(file_path)} - skipping.")
            return None
        except Exception as e:
            print(f"Load error for {os.path.basename(file_path)}: {e}")
            return None

    def _load_and_start(self, selected):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._load_single_track, file_path) for file_path in selected]
            new_tracks = [f.result() for f in futures if f.result() is not None]
        
        if not new_tracks:
            self.after(0, lambda: self.status_label.config(text="Load failed for all tracks."))
            self.after(0, self.stop_mix)
            return
            
        if self.is_ambient_mode.get() is False:
            for track in new_tracks:
                track.is_waiting_for_sync = True 

        with self.data_lock:
            self.active_tracks = new_tracks
            for widget in self.tracks_frame.winfo_children():
                widget.destroy()
            self.track_frames.clear()
        
        self.after(0, self._setup_audio)
        self.after(0, self._display_tracks)
    
    def _setup_audio(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping old stream: {e}")
            
        try:
            self.stream = sd.OutputStream(
                samplerate=self.RATE,
                channels=self.CHANNELS,
                blocksize=self.CHUNK,
                callback=self.audio_callback,
                finished_callback=self._on_audio_finished,
                dtype='float32'
            )
            self.stream.start()
            self.status_label.config(text=f"Playing {len(self.active_tracks)} tracks")
        except Exception as e:
            self.status_label.config(text=f"Audio error: {e}")
            self.stop_mix()
    
    def audio_callback(self, outdata, frames, time_info, status):
        
        if self.is_ambient_mode_cache is False:
            self.master_sample_counter += frames
        
        mixed = np.zeros((frames, 2), dtype=np.float32)
        mode = 'ambient' if self.is_ambient_mode_cache else 'sync'
        
        try:
            current_params_items = list(self.track_params.items())
        except RuntimeError:
            current_params_items = [] 
            
        for track, params in current_params_items:
            if track in self.active_tracks:
                chunk = track.get_chunk(frames, params, mode)
                
                if chunk is not None:
                    mixed += chunk
        
        mixed *= self.master_volume_cache
        
        if self.effects_dirty or self.reverb_cache > 0.005 or self.delay_cache > 0.005 or self.drive_cache > 0.005:
            try:
                mixed = self.master_effects(mixed, self.RATE, reset=False)
                self.effects_dirty = False
            except Exception as e:
                print(f"Effects error: {e}")
        
        mixed = np.clip(mixed, -1.0, 1.0)
        outdata[:] = mixed
        
        # --- NEW: Capture for recording/export ---
        # Convert float32 to int16 bytes
        rec_chunk = (mixed * 32767).astype(np.int16)
        rec_bytes = rec_chunk.tobytes()
        
        if self.recording_event.is_set():
            self.record_frames.append(rec_bytes)
            
        if self.exporting:
            self.export_audio_frames.append(rec_bytes)
        # --- END NEW ---
    
    def _on_audio_finished(self):
        print("Audio stream finished unexpectedly.")
        if self.audio_active.is_set():
            self.stop_mix()
    
    def _display_tracks(self):
        with self.data_lock:
            for widget in self.tracks_frame.winfo_children():
                widget.destroy()
            self.track_frames.clear()
            
            for i, track in enumerate(self.active_tracks):
                frame = tk.Frame(self.tracks_frame, bg='lightblue')
                frame.pack(fill=tk.X, pady=2, padx=5)
                
                label = tk.Label(frame, text=f"{i+1}. {track.name}", bg='lightblue', width=30, anchor='w')
                label.pack(side=tk.LEFT)
                
                vol_slider = tk.Scale(frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, 
                        variable=track.base_volume, length=150, showvalue=True)
                vol_slider.pack(side=tk.LEFT)
                vol_slider.config(command=lambda *args: self._on_param_change())
                
                speed_slider = tk.Scale(frame, from_=0.5, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, 
                        variable=track.speed_var, length=100, showvalue=True)
                speed_slider.pack(side=tk.LEFT, padx=5)
                speed_slider.config(command=lambda *args: self._on_param_change())
                
                pan_slider = tk.Scale(frame, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, 
                        variable=track.base_pan, length=100, showvalue=True)
                pan_slider.pack(side=tk.LEFT)
                pan_slider.config(command=lambda *args: self._on_param_change())
                
                progress = ttk.Progressbar(frame, length=100, mode='determinate', 
                              variable=track.progress_var)
                progress.pack(side=tk.LEFT, padx=5)
                
                self.track_frames[track] = {
                    'frame': frame, 'label': label, 'vol_slider': vol_slider,
                    'speed_slider': speed_slider, 'pan_slider': pan_slider,
                    'progress': progress
                }
        
        self.update_progress()
    
    def ambient_digger_thread(self):
        last_swap_time = time.time()
        preload_retries = 0
        MAX_RETRIES = 5
        
        while self.audio_active.is_set():
            
            if self.is_ambient_mode_cache and not self.is_currently_swapping:
                density = self.ambient_density_var.get()
                
                if time.time() - last_swap_time > density:
                    if self.next_track_buffer is not None:
                        self.is_currently_swapping = True 
                        
                        with self.data_lock:
                            if not self.active_tracks:
                                self.is_currently_swapping = False
                                continue
                            track_to_replace = random.choice(self.active_tracks)
                        
                        new_track = self.next_track_buffer
                        self.next_track_buffer = None
                        
                        print(f"--- INSTANT SWAP: {new_track.name}")
                        self.after(0, self._finalize_swap, track_to_replace, new_track)
                        
                        last_swap_time = time.time()
                        self.preload_event.set() 
                        preload_retries = 0
                    else:
                        print("Waiting for pre-load...")
            
            if self.preload_event.is_set() and self.next_track_buffer is None:
                self.preload_event.clear()
                
                with self.data_lock:
                    active_names = [t.name for t in self.active_tracks]
                    available_files = [f for f in self.all_audio_files if os.path.basename(f)[:30] not in active_names]
                
                if available_files and preload_retries < MAX_RETRIES:
                    new_file = random.choice(available_files)
                    
                    def load_preload_track():
                        loaded_track = self._load_single_track(new_file)
                        if loaded_track is not None:
                            self.next_track_buffer = loaded_track
                            print(f"+++ Pre-loaded: {loaded_track.name}")
                        else:
                            print(f"Pre-load failed for {os.path.basename(new_file)}, retrying...")
                            self.preload_event.set()
                    
                    threading.Thread(target=load_preload_track, daemon=True).start()
                    preload_retries += 1
                else:
                    if not available_files:
                        print("No available files to preload.")
                    if preload_retries >= MAX_RETRIES:
                        print("Max preload retries reached—pausing preloads.")
                    
            time.sleep(1.0)

    def _finalize_swap(self, old_track, new_track):
        if not self.audio_active.is_set(): 
            self.is_currently_swapping = False
            return
            
        print(f"Finalizing swap: {old_track.name} -> {new_track.name}")
        
        with self.data_lock:
            try:
                idx = self.active_tracks.index(old_track)
                self.active_tracks[idx] = new_track
                
                ui_elements = self.track_frames.get(old_track)
                if ui_elements:
                    ui_elements['vol_slider'].config(variable=new_track.base_volume, command=lambda *args: self._on_param_change())
                    ui_elements['speed_slider'].config(variable=new_track.speed_var, command=lambda *args: self._on_param_change())
                    ui_elements['pan_slider'].config(variable=new_track.base_pan, command=lambda *args: self._on_param_change())
                    ui_elements['progress'].config(variable=new_track.progress_var)
                    ui_elements['label'].config(text=f"{idx+1}. {new_track.name}")
                    
                    self.track_frames[new_track] = self.track_frames.pop(old_track)
                
            except ValueError:
                print("Track already gone, swap cancelled.")
            except Exception as e:
                print(f"Error finalizing swap: {e}")
            finally:
                self.is_currently_swapping = False

    def update_slider_cache(self):
        if not self.audio_active.is_set():
            return 

        new_slider_cache = {}
        
        with self.data_lock: 
            current_tracks = self.active_tracks[:]
            
        for track in current_tracks:
            try:
                new_slider_cache[track] = {
                    'vol': track.base_volume.get(),
                    'pan': track.base_pan.get(),
                    'speed': track.speed_var.get()
                }
            except tk.TclError:
                pass
        
        with self.data_lock:
            self.slider_cache = new_slider_cache

        self.after(50, self.update_slider_cache)

    def cache_params_thread(self):
        lfo_pan_mod = {}
        lfo_vol_mod = {}
        
        while self.audio_active.is_set():
            self.param_update_event.wait(0.1)
            self.param_update_event.clear()
            
            now = time.time()
            
            self.master_volume_cache = self.master_volume.get()
            new_reverb = self.reverb_var.get()
            new_delay = self.delay_var.get()
            new_drive = self.drive_var.get()
            self.bpm_cache = self.bpm_var.get()
            self.is_ambient_mode_cache = self.is_ambient_mode.get()
            
            if abs(new_reverb - self.reverb_cache) > 0.001:
                self.reverb.wet_level = new_reverb * 0.5
                self.reverb_cache = new_reverb
                self.effects_dirty = True
            
            if abs(new_delay - self.delay_cache) > 0.001:
                self.delay.mix = new_delay * 0.5
                self.delay_cache = new_delay
                self.effects_dirty = True
            
            if abs(new_drive - self.drive_cache) > 0.001:
                self.distortion.drive_db = new_drive * 40.0
                self.drive_cache = new_drive
                self.effects_dirty = True
            
            with self.data_lock:
                current_slider_values = self.slider_cache.copy()

            new_params = {} 
            
            for track, slider_vals in current_slider_values.items():
                
                if self.is_ambient_mode_cache:
                    if track not in lfo_pan_mod: 
                        lfo_pan_mod[track] = 0.0
                        lfo_vol_mod[track] = 1.0
                        
                    lfo_pan_mod[track] = math.sin(now * 0.1 + (hash(track) % 100)) * 0.2
                    lfo_vol_mod[track] = 0.85 + (math.sin(now * 0.07 + (hash(track) % 50)) * 0.15)
                    
                    vol = slider_vals['vol'] * lfo_vol_mod[track]
                    pan = slider_vals['pan'] + lfo_pan_mod[track]
                else:
                    vol = slider_vals['vol']
                    pan = slider_vals['pan']
                
                new_params[track] = (
                    slider_vals['speed'],
                    np.clip(vol, 0.0, 1.0),
                    np.clip(pan, -1.0, 1.0)
                )

            self.track_params = new_params
            
            if self.is_ambient_mode_cache is False:
                samples_per_bar = self.calculate_samples_per_bar(self.bpm_cache)
                if self.master_sample_counter >= samples_per_bar:
                    self.master_sample_counter %= samples_per_bar
                    
                    with self.data_lock:
                        for track in self.active_tracks:
                            if track.is_waiting_for_sync:
                                track.is_waiting_for_sync = False
                                track.pointer = 0
    
    def update_progress(self):
        if self.audio_active.is_set():
            with self.data_lock:
                for track in self.active_tracks:
                    if track in self.track_frames: 
                        if track.length > 0 and not track.is_waiting_for_sync:
                            track.progress_var.set((track.pointer / track.length) * 100)
                        elif track.is_waiting_for_sync:
                            track.progress_var.set(100)
            self.after(500, self.update_progress)
    
    def stop_mix(self):
        print("--- STOP MIX ---")
        self.audio_active.clear() 
        self.recording_event.clear()
        
        # --- NEW: Stop visual thread ---
        self.stop_visual_thread() # This now blocks for up to 2.0s
        # --- END NEW ---
        
        self.preload_event.clear()
        self.next_track_buffer = None
        
        if self.cache_thread and self.cache_thread.is_alive():
            print("Waiting for cache thread...")
            self.cache_thread.join(timeout=1.0) # Increased
        if self.digger_thread and self.digger_thread.is_alive():
            print("Waiting for digger thread...")
            self.digger_thread.join(timeout=1.5) # Increased
            
        print("Audio/Digger threads stopped.")
            
        self.play_button.config(text="▶ PLAY", bg='green')
        self.record_button.config(text="🔴 RECORD Audio", bg='darkred')
        self.status_label.config(text="Stopped")
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        with self.data_lock:
            self.active_tracks = []
            self.track_params.clear()
            self.slider_cache.clear()
            for widget in self.tracks_frame.winfo_children():
                widget.destroy()
            self.track_frames.clear()
        print("--- STOP MIX COMPLETE ---")
    
    def start_new_mix(self):
        print("Start new mix pressed...")
        
        # --- NEW: Disable button to prevent double-clicks ---
        self.next_button_widget = None
        try:
            # Find the 'NEXT' button to disable it
            for child in self.right_ctrl_frame.winfo_children():
                if 'mix_btn_frame' in str(child):
                    for btn in child.winfo_children():
                        if btn.cget('text') == "⏭ NEXT":
                            self.next_button_widget = btn # Save for re-enabling
                            btn.config(state=tk.DISABLED, text="...")
                            break
                    if self.next_button_widget:
                        break
        except Exception as e:
            print(f"Could not disable NEXT button: {e}")

        # --- NEW: Run stop/start in a separate thread to avoid UI freeze ---
        threading.Thread(target=self._restart_mix_thread, daemon=True).start()

    def _restart_mix_thread(self):
        """Worker thread to safely stop and restart the mix."""
        print("Restart thread: Calling stop_mix()...")
        if self.audio_active.is_set():
            # This is synchronous and will block THIS thread, not the UI
            self.stop_mix() 
            print("Restart thread: stop_mix() finished.")
        else:
            print("Restart thread: Audio already stopped.")

        # Now, tell the main thread to start the mix again
        # This will load new tracks and a new visual
        self.after(0, self._start_mix_from_thread)

    def _start_mix_from_thread(self):
        """Callback to start mix from the main UI thread."""
        print("Main thread: Calling start_mix()...")
        self.start_mix() # This loads new tracks and calls load_next_visual()
        
        # Re-enable the NEXT button
        if self.next_button_widget:
            try:
                self.next_button_widget.config(state=tk.NORMAL, text="⏭ NEXT")
            except tk.TclError:
                pass # Window might be closing
        print("Main thread: Mix restarted.")
    
    # --- CONTROLS ---
    def toggle_mode(self):
        new_mode = not self.is_ambient_mode.get()
        self.is_ambient_mode.set(new_mode)
        self.update_ui_mode()
        
    def update_ui_mode(self):
        if self.is_ambient_mode.get():
            self.mode_button.config(text="Mode: AMBIENT", bg='orange')
            self.bpm_slider.config(state=tk.DISABLED)
            self.density_slider.config(state=tk.NORMAL)
            self.status_label.config(text="Ambient Mode")
            self.preload_event.set()
        else:
            self.mode_button.config(text="Mode: SYNC", bg='cyan')
            self.bpm_slider.config(state=tk.NORMAL)
            self.density_slider.config(state=tk.DISABLED)
            self.status_label.config(text="Sync Mode")
            self.preload_event.clear()
            self.next_track_buffer = None
    
    def randomize_volumes(self):
        if not self.active_tracks:
            return
        with self.data_lock:
            for track in self.active_tracks:
                track.base_volume.set(round(random.uniform(0.3, 1.0), 2))
        self.status_label.config(text="Volumes randomized!")
    
    def reset_speeds_pans(self):
        if not self.active_tracks:
            return
        with self.data_lock:
            for track in self.active_tracks:
                track.speed_var.set(1.0)
                track.base_pan.set(0.0)
        self.status_label.config(text="Reset!")
    
    # --- Smart Filename Logic ---
    def _clean_filename(self, name):
        name = name.lower()
        name = re.sub(r'\.wav|\.mp3|\.flac|\.m4a|\.ogg', '', name, flags=re.IGNORECASE)
        name = re.sub(r'[^\w\u0600-\u06FF]+', '_', name) 
        return name.strip('_')[:15] 

    def toggle_record(self):
        if self.exporting:
            messagebox.showwarning("Busy", "Cannot record audio while exporting video.")
            return
            
        if not self.audio_active.is_set():
            self.status_label.config(text="Start playing first!")
            return
        
        if not self.recording_event.is_set():
            
            with self.data_lock:
                track_names = [t.name for t in self.active_tracks]
            
            clean_names = [self._clean_filename(name) for name in track_names]
            name_part = "_".join(clean_names)
            if not name_part: name_part = "mix"
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            default_filename = f"al_hut_{name_part}_{timestamp}.wav"
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav")],
                initialfile=default_filename
            )
            if filename:
                self.record_frames.clear()
                self.record_filename = filename
                self.after(50, lambda: self.recording_event.set())
                self.record_button.config(text="■ STOP", bg='orange')
                self.status_label.config(text="Recording...")
        else:
            self.recording_event.clear()
            self.record_button.config(text="🔴 RECORD Audio", bg='darkred')
            threading.Thread(target=self._save_recording, daemon=True).start()
    
    def _save_recording(self):
        if not self.record_frames:
            self.status_label.config(text="No audio recorded")
            return
        
        try:
            with wave.open(self.record_filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.record_frames))
            
            duration = (len(self.record_frames) * self.CHUNK) / self.RATE
            self.status_label.config(text=f"Saved {int(duration)}s audio")
            messagebox.showinfo("Success", f"Audio saved!\nDuration: {int(duration)}s")
        except Exception as e:
            self.status_label.config(text=f"Save error: {e}")
        finally:
            self.record_frames.clear()
            self.record_filename = None # Clear filename after saving
    
    # --- NEW EXPORT FUNCTIONS (from mixer69) ---
    def toggle_export(self):
        if self.recording_event.is_set():
            messagebox.showwarning("Busy", "Cannot export video while recording audio.")
            return
        
        if not self.audio_active.is_set():
            messagebox.showerror("Error", "Start playing the mix before exporting.")
            return
            
        if not self.exporting:
            output_filename = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4")],
                initialfile=f"al_hut_export_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
                title="Save Video Export"
            )
            if not output_filename:
                return

            # Check for ffmpeg
            if not shutil.which('ffmpeg'):
                messagebox.showerror("Error", "ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH to export videos.")
                return

            self.export_output_filename = output_filename
            self.exporting = True
            self.export_video_frames = []
            self.export_audio_frames = []
            self.export_start_time = time.time()
            
            self.export_button.config(text="FINISH EXPORT", bg='orange')
            self.status_label.config(text="Exporting video...")
            self.update_export_timer()
            
            self.export_thread = threading.Thread(target=self._video_capture_loop, daemon=True)
            self.export_thread.start()
        else:
            self.exporting = False
            self.status_label.config(text="Finalizing export... Please wait.")
            self.export_button.config(text="📹 Export A/V", bg='purple', state=tk.DISABLED)
            
            if self.export_timer:
                self.after_cancel(self.export_timer)
            self.export_time_label.config(text="")

            time.sleep(0.5) # Give capture thread a moment to finish
            
            threading.Thread(target=self._combine_av, daemon=True).start()

    def update_export_timer(self):
        if self.exporting:
            elapsed = int(time.time() - self.export_start_time)
            mins, secs = divmod(elapsed, 60)
            self.export_time_label.config(text=f"EXPORT: {mins:02d}:{secs:02d}")
            self.export_timer = self.after(1000, self.update_export_timer)

    def _video_capture_loop(self):
        FPS = 30 
        while self.exporting:
            frame_to_save = None
            with self.current_frame_lock:
                if self.current_pil_image:
                    # Convert PIL (RGB) to OpenCV (BGR)
                    frame = cv2.cvtColor(np.array(self.current_pil_image), cv2.COLOR_RGB2BGR)
                    frame_to_save = frame
            
            if frame_to_save is not None:
                self.export_video_frames.append(frame_to_save)
                
            time.sleep(1 / FPS)
    
    def _combine_av(self):
        if not self.export_video_frames or not self.export_audio_frames:
            self.after(0, lambda: messagebox.showerror("Error", "No video or audio data captured for export."))
            self.after(0, self.export_button.config, {'state': tk.NORMAL})
            return

        temp_audio_file = 'temp_export_audio.wav'
        temp_video_file = 'temp_export_video.avi'

        try:
            with wave.open(temp_audio_file, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.export_audio_frames))
            print("Temp audio saved.")
        except Exception as e:
            print(f"Error saving temp audio: {e}")
            self.after(0, self.export_button.config, {'state': tk.NORMAL})
            return

        try:
            # Ensure dimensions are even for ffmpeg
            height, width, _ = self.export_video_frames[0].shape
            if width % 2 != 0: width -= 1
            if height % 2 != 0: height -= 1
                
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_video_file, fourcc, 30.0, (width, height))
            for frame in self.export_video_frames:
                # Resize frame to even dimensions if needed
                resized_frame = cv2.resize(frame, (width, height))
                out.write(resized_frame)
            out.release()
            print("Temp video saved.")
        except Exception as e:
            print(f"Error saving temp video: {e}")
            self.after(0, self.export_button.config, {'state': tk.NORMAL})
            return
            
        try:
            print("Starting ffmpeg combine...")
            cmd = [
                'ffmpeg', '-y', 
                '-i', temp_video_file, 
                '-i', temp_audio_file, 
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '19',
                '-c:a', 'aac', '-b:a', '192k',
                '-shortest', self.export_output_filename
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            print("ffmpeg combine complete.")
            self.after(0, lambda: messagebox.showinfo("Success", f"Video exported successfully to {self.export_output_filename}"))
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg error: {e.stderr}")
            self.after(0, lambda: messagebox.showerror("ffmpeg Error", f"Failed to combine video and audio.\n\nError:\n{e.stderr}"))
        except Exception as e:
            print(f"Combine error: {e}")
            self.after(0, lambda: messagebox.showerror("Error", f"An unexpected error occurred during export: {e}"))
        finally:
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
            if os.path.exists(temp_video_file):
                os.remove(temp_video_file)

            self.export_video_frames = []
            self.export_audio_frames = []
            self.after(0, self.export_button.config, {'state': tk.NORMAL})
            self.after(0, self.status_label.config, {'text': 'Export finished.'})
    # --- END NEW EXPORT FUNCTIONS ---


    # --- Visual Playback Functions ---
    def load_next_visual(self):
        if not self.all_visual_files or not VJ_LIBS_INSTALLED:
            return
        
        try:
            self.current_visual_path = random.choice(self.all_visual_files)
            print(f"Loading visual: {os.path.basename(self.current_visual_path)}")
            
            self.stop_visual_thread()
            
            self.visual_stop_event.clear()
            self.visual_thread = threading.Thread(target=self.visual_playback_loop, 
                                                  args=(self.current_visual_path,), 
                                                  daemon=True)
            self.visual_thread.start()
        except Exception as e:
            print(f"Error loading next visual: {e}")

    def stop_visual_thread(self):
        print("Attempting to stop visual thread...")
        if self.visual_thread and self.visual_thread.is_alive():
            self.visual_stop_event.set()
            print("Visual stop event set. Joining thread (timeout 2.0s)...")
            self.visual_thread.join(timeout=2.0) # Increased timeout
            
            if self.visual_thread.is_alive():
                print("WARNING: Visual thread did not join in time.")
            else:
                print("Visual thread joined successfully.")
                
            del self.visual_thread  # Help GC
            self.visual_thread = None
        else:
            print("Visual thread already stopped or not running.")
        
        # Clear the label from the main thread
        self.after(0, self.clear_visual_label)
        print("Visual thread stop complete.")


    def visual_playback_loop(self, file_path):
        """Dispatches to the correct playback function based on extension."""
        if not file_path:
            return
            
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext in ['.jpg', '.jpeg', '.png']:
                self.play_static_image(file_path)
            elif ext == '.gif':
                self.play_gif(file_path)
            elif ext in ['.mp4', '.mkv', '.avi', '.webm']:
                self.play_video(file_path)
        except Exception as e:
            print(f"Error in playback loop for {os.path.basename(file_path)}: {e}")
            self.after(0, self.clear_visual_label) # Bad screen on error

    def resize_image(self, img):
        """Resizes a PIL image to fit the visual_label, preserving aspect ratio."""
        box_w = self.visual_label_width
        box_h = self.visual_label_height
        
        if box_w < 2 or box_h < 2: 
            box_w, box_h = 640, 360

        img_w, img_h = img.size
        if img_w == 0 or img_h == 0:
            return img # Bad image

        scale = min(box_w / img_w, box_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # --- OPTIMIZED: Use faster resizing algorithm ---
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _safe_update_visual_label(self, photo, img_for_export):
        """Helper function to safely update the image on the main thread."""
        try:
            self.visual_label.config(image=photo)
            self.visual_label.image = photo # Keep reference to prevent GC
            
            # Save frame for export thread
            with self.current_frame_lock:
                self.current_pil_image = img_for_export
                
        except tk.TclError:
            pass # App closing
            
    def play_static_image(self, file_path):
        """Displays a single static image."""
        if self.visual_stop_event.is_set(): return
        img = Image.open(file_path)
        img_resized = self.resize_image(img.convert('RGBA'))
        photo = ImageTk.PhotoImage(img_resized)
        
        # --- MODIFIED: Pass both photo (for display) and img (for export) ---
        self.after(0, self._safe_update_visual_label, photo, img_resized.convert('RGB'))

    def play_gif(self, file_path):
        """Loops through a GIF's frames."""
        gif = Image.open(file_path)
        frames_display = [] # For Tkinter
        frames_export = []  # For export (RGB)
        
        try:
            duration = gif.info.get('duration', 40) / 1000.0 # to seconds
        except:
            duration = 0.04 # default to 25fps
            
        for frame in ImageSequence.Iterator(gif):
            if self.visual_stop_event.is_set(): return
            
            img = frame.copy().convert('RGBA')
            img_resized = self.resize_image(img)
            
            frames_display.append(ImageTk.PhotoImage(img_resized))
            frames_export.append(img_resized.convert('RGB'))
        
        if not frames_display: return
        
        idx = 0
        while not self.visual_stop_event.is_set():
            frame_photo = frames_display[idx]
            frame_img = frames_export[idx]
            
            self.after(0, self._safe_update_visual_label, frame_photo, frame_img)
            
            idx = (idx + 1) % len(frames_display)
            time.sleep(duration)

    def play_video(self, file_path):
        """Loops a video using OpenCV."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {file_path}")
            return
            
        fps = min(cap.get(cv2.CAP_PROP_FPS), 30)  # Cap at 30 FPS to reduce CPU
        frame_delay = 1.0 / fps if fps > 0 else 0.04
        
        while not self.visual_stop_event.is_set():
            frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                continue
            
            # Convert frame for PIL/Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            img_resized = self.resize_image(img)
            photo = ImageTk.PhotoImage(img_resized)
            
            self.after(0, self._safe_update_visual_label, photo, img_resized)
            
            # Frame skipping if delayed
            elapsed = time.time() - frame_start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap.release()

    def clear_visual_label(self):
        """Clears the visual display area."""
        try:
            self.visual_label.config(image=None)
            self.visual_label.image = None
            with self.current_frame_lock:
                self.current_pil_image = None
        except tk.TclError:
            pass # App closing
    
    def on_closing(self):
        print("--- ON CLOSING ---")
        self.visual_stop_event.set() # Signal visual thread to stop
        
        if self.exporting:
            self.exporting = False # Stop export
            # We don't save, just stop
            print("Stopping export...")
        if self.export_timer:
            self.after_cancel(self.export_timer)
            
        self.stop_mix() # This will stop audio threads
        
        # Explicitly join visual thread one last time
        if self.visual_thread and self.visual_thread.is_alive():
            print("Waiting for visual thread on close...")
            self.visual_thread.join(timeout=1.0)
            
        if self.export_thread and self.export_thread.is_alive():
            print("Waiting for export thread on close...")
            self.export_thread.join(timeout=1.0)
            
        print("All threads stopped. Destroying window.")
        self.destroy()

if __name__ == "__main__":
    app = AlHutMixer()
    app.update_idletasks() # Ensure window is drawn
    # Get initial size after window is drawn
    app.visual_label_width = app.visual_label.winfo_width()
    app.visual_label_height = app.visual_label.winfo_height()
    
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()