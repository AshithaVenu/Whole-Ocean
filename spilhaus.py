"""
Spilhaus SST map -> continent labels -> zoom into the Antarctic Circumpolar Current
with flowing particles colored by SST.

Setup (clean env recommended):
  conda create -n spilhaus -c conda-forge python=3.11 xarray netcdf4 h5netcdf h5py pyproj matplotlib scipy ffmpeg
  conda activate spilhaus
  pip install copernicusmarine
  copernicusmarine login

Run:  python spilhaus_video_v2.py
Set PREVIEW = True first for a fast, low-res test render.
"""
import os, datetime as dt, urllib.request, urllib.error
import numpy as np, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from pyproj import Transformer

# ---------------- config ----------------
PREVIEW = True            # True = 540px, 15 fps (fast). False = 1080px, 30 fps.
OUT = "spilhaus_acc.mp4"
CURRENTS_FILE = "currents.nc"
CMEMS_ID = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"

PX = 540 if PREVIEW else 1080
FPS = 15 if PREVIEW else 30
DURATION = 24             # seconds
N_PART = 5000 if PREVIEW else 8000
TRAIL = 12                # trail length in frames
TIME_SCALE = 2.0          # simulated days of flow per video frame
LAT_MIN, LAT_MAX = -75, -30   # where particles live
CMAP = plt.get_cmap("turbo")
NORM = colors.Normalize(vmin=-2, vmax=28)
BG = "#0b1020"
R_EARTH = 6371000.0
DEG = 0.25

# ---------------- data ----------------
BASE = ("https://www.ncei.noaa.gov/data/sea-surface-temperature-"
        "optimum-interpolation/v2.1/access/avhrr")

def fetch_sst_file(start_back=2, tries=12):
    for d in range(start_back, start_back + tries):
        day = dt.date.today() - dt.timedelta(days=d)
        ym, ymd = day.strftime("%Y%m"), day.strftime("%Y%m%d")
        for suffix in ("", "_preliminary"):
            name = f"oisst-avhrr-v02r01.{ymd}{suffix}.nc"
            if os.path.exists(name):
                return name
            try:
                urllib.request.urlretrieve(f"{BASE}/{ym}/{name}", name)
                print("Downloaded", name)
                return name
            except urllib.error.HTTPError:
                continue
    raise RuntimeError("No OISST file found")

def load_sst():
    sst = xr.open_dataset(fetch_sst_file()).sst.squeeze(drop=True)
    sst = sst.assign_coords(lon=((sst.lon + 180) % 360) - 180).sortby("lon")
    return sst

def load_currents(sst):
    if not os.path.exists(CURRENTS_FILE):
        import copernicusmarine as cm
        today = dt.date.today()
        cm.subset(
            dataset_id=CMEMS_ID, variables=["uo", "vo"],
            minimum_longitude=-180, maximum_longitude=179.9,
            minimum_latitude=-80, maximum_latitude=-25,
            minimum_depth=0.494, maximum_depth=0.495,
            start_datetime=(today - dt.timedelta(days=6)).isoformat(),
            end_datetime=(today - dt.timedelta(days=3)).isoformat(),
            output_filename=CURRENTS_FILE,
        )
    ds = xr.open_dataset(CURRENTS_FILE)
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = ds.isel(time=0).squeeze(drop=True)
    if "depth" in ds.dims:
        ds = ds.isel(depth=0)
    lats = sst.lat.values[(sst.lat.values >= -80) & (sst.lat.values <= -25)]
    ds = ds.interp(lat=lats, lon=sst.lon.values, method="linear")
    return lats, ds.uo.values, ds.vo.values

# ---------------- projection ----------------
def make_transformer():
    candidates = [
        "+proj=spilhaus +R=6371007",
        ("+proj=ob_tran +o_proj=adams_ws2 +o_lat_c=-66.94970198 "
         "+o_lon_c=49.56371678 +o_alpha=40.17823482 +R=6371007"),
    ]
    for p in candidates:
        try:
            t = Transformer.from_crs("EPSG:4326", p, always_xy=True)
            x, y = t.transform(0, -60)
            if np.isfinite(x) and np.isfinite(y):
                print("Using projection:", p)
                return t
        except Exception as e:
            print("Failed:", p, "->", e)
    raise RuntimeError("No working projection")

def build_base(tf, sst, n=1400):
    lo, la = np.meshgrid(sst.lon.values, sst.lat.values)
    x, y = tf.transform(lo, la)
    ok = np.isfinite(x) & np.isfinite(y)
    cx = (x[ok].min() + x[ok].max()) / 2
    cy = (y[ok].min() + y[ok].max()) / 2
    h = max(x[ok].max() - x[ok].min(), y[ok].max() - y[ok].min()) / 2 * 1.01
    gx = np.linspace(cx - h, cx + h, n)
    gy = np.linspace(cy + h, cy - h, n)   # top-down for imshow
    X, Y = np.meshgrid(gx, gy)
    glon, glat = tf.transform(X, Y, direction="INVERSE")
    valid = np.isfinite(glon) & np.isfinite(glat)
    iy = np.clip(np.round((np.where(valid, glat, 0) - sst.lat.values[0]) / DEG).astype(int), 0, len(sst.lat) - 1)
    ix = np.round((np.where(valid, glon, 0) - sst.lon.values[0]) / DEG).astype(int) % len(sst.lon)
    vals = sst.values[iy, ix]
    vals = np.where(valid, vals, np.nan)
    rgba = CMAP(NORM(vals))
    rgba[~np.isfinite(vals)] = (0, 0, 0, 0)   # land / outside -> transparent
    return rgba, (cx - h, cx + h, cy - h, cy + h), (cx, cy, h)

# ---------------- particles ----------------
class Flow:
    def __init__(self, tf, sst, lats, U, V, rng):
        self.tf, self.rng = tf, rng
        self.sst = sst.values
        self.slat0, self.slon0 = sst.lat.values[0], sst.lon.values[0]
        self.lat0, self.lon0 = lats[0], sst.lon.values[0]
        self.U, self.V = U, V
        self.ny, self.nx = U.shape
        self.lon = np.zeros(N_PART); self.lat = np.zeros(N_PART)
        self.age = np.zeros(N_PART)
        self.maxage = rng.integers(60, 140, N_PART)
        self.respawn(np.ones(N_PART, bool))
        x, y = tf.transform(self.lon, self.lat)
        self.hist = np.repeat(np.stack([x, y], -1)[None], TRAIL, axis=0)

    def _idx(self, lon, lat):
        iy = np.clip(np.round((lat - self.lat0) / DEG).astype(int), 0, self.ny - 1)
        ix = np.round((lon - self.lon0) / DEG).astype(int) % self.nx
        return iy, ix

    def respawn(self, m):
        for _ in range(6):   # retry until on ocean
            k = m.sum()
            if k == 0:
                break
            self.lon[m] = self.rng.uniform(-180, 180, k)
            self.lat[m] = self.rng.uniform(LAT_MIN, LAT_MAX, k)
            iy, ix = self._idx(self.lon, self.lat)
            m = m & ~np.isfinite(self.U[iy, ix])
        self.age[m] = 0

    def step(self):
        iy, ix = self._idx(self.lon, self.lat)
        u, v = self.U[iy, ix], self.V[iy, ix]
        dead = ~np.isfinite(u) | ~np.isfinite(v)
        u = np.nan_to_num(u); v = np.nan_to_num(v)
        dt_s = 86400.0 * TIME_SCALE
        self.lat += np.degrees(v * dt_s / R_EARTH)
        self.lon += np.degrees(u * dt_s / (R_EARTH * np.cos(np.radians(self.lat))))
        self.lon = (self.lon + 180) % 360 - 180
        self.age += 1
        dead |= (self.age > self.maxage) | (self.lat < -85) | (self.lat > -20)
        newpos = dead.copy()
        if dead.any():
            self.age[dead] = 0
            self.respawn(dead)
        x, y = self.tf.transform(self.lon, self.lat)
        xy = np.stack([x, y], -1)
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = xy
        self.hist[:, newpos] = xy[newpos]    # no trail across a respawn

    def sst_at(self):
        iy = np.clip(np.round((self.lat - self.slat0) / DEG).astype(int), 0, self.sst.shape[0] - 1)
        ix = np.round((self.lon - self.slon0) / DEG).astype(int) % self.sst.shape[1]
        return self.sst[iy, ix]

# ---------------- helpers ----------------
def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

LABELS = [  # name, lon, lat
    ("ANTARCTICA", 0, -88),
    ("AFRICA", 20, 3),
    ("EUROPE", 15, 52),
    ("ASIA", 90, 45),
    ("NORTH AMERICA", -100, 45),
    ("SOUTH AMERICA", -60, -15),
    ("AUSTRALIA", 134, -25),
]

# ---------------- main ----------------
def main():
    rng = np.random.default_rng(1)
    sst = load_sst()
    lats, U, V = load_currents(sst)
    tf = make_transformer()
    rgba, extent, (cx, cy, h) = build_base(tf, sst)
    flow = Flow(tf, sst, lats, U, V, rng)

    # zoom target: ring at 35S around the pole (centre of the map)
    px0, py0 = tf.transform(0, -90)
    ring = np.array([tf.transform(l, -35) for l in np.linspace(-180, 180, 73)])
    zoom_half = float(np.nanmax(np.hypot(ring[:, 0] - px0, ring[:, 1] - py0))) * 1.08
    jump = 0.05 * h

    fig = plt.figure(figsize=(PX / 100, PX / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG); ax.axis("off")
    ax.set_aspect("equal")
    base = ax.imshow(rgba, extent=extent, zorder=1, interpolation="bilinear")
    lc = LineCollection([], linewidths=0.9 if PREVIEW else 1.3, zorder=3)
    ax.add_collection(lc)

    stroke = [pe.withStroke(linewidth=3, foreground="black")]
    texts = []
    for i, (name, lo, la) in enumerate(LABELS):
        x, y = tf.transform(lo, la)
        t = ax.text(x, y, name, color="white", ha="center", va="center",
                    fontsize=9 if PREVIEW else 15, fontweight="bold",
                    path_effects=stroke, alpha=0, zorder=4)
        texts.append((t, 4.0 + 0.55 * i))

    title = fig.text(0.5, 0.945, "Antarctic Circumpolar Current", color="white", ha="center",
                     fontsize=13 if PREVIEW else 26, fontweight="bold", alpha=0, zorder=5)
    sub = fig.text(0.5, 0.905, "The world's strongest ocean current, colored by sea surface temperature",
                   color="#cfd6e6", ha="center", fontsize=7 if PREVIEW else 13, alpha=0, zorder=5)

    # ---- legend (top right) with a faint backing panel ----
    fig.add_artist(Rectangle((0.885, 0.57), 0.11, 0.36, transform=fig.transFigure,
                             facecolor="black", alpha=0.4, edgecolor="none", zorder=4))
    cax = fig.add_axes([0.93, 0.60, 0.016, 0.28])      # x, y, width, height (figure fraction)
    cax.set_zorder(5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="vertical")
    cb.set_label("SST (°C)", color="white", fontsize=7 if PREVIEW else 12)
    cax.yaxis.set_label_position("left")
    cax.yaxis.set_ticks_position("right")
    cb.ax.tick_params(colors="white", labelsize=6 if PREVIEW else 10)
    cb.outline.set_edgecolor("#555")

    writer = animation.FFMpegWriter(fps=FPS, codec="libx264",
                                    extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    ramp = np.linspace(0, 1, TRAIL - 1) ** 1.6
    total = DURATION * FPS
    with writer.saving(fig, OUT, dpi=100):
        for f in range(total):
            t = f / FPS
            flow.step()

            zoom = smooth((t - 10) / 4)                   # 0 -> 1 between 10s and 14s
            half = (1 - zoom) * h + zoom * zoom_half
            ccx = (1 - zoom) * cx + zoom * px0
            ccy = (1 - zoom) * cy + zoom * py0
            ax.set_xlim(ccx - half, ccx + half)
            ax.set_ylim(ccy - half, ccy + half)

            base.set_alpha(smooth(t / 2.5) * (1 - 0.7 * zoom))

            for txt, t0 in texts:
                txt.set_alpha(smooth((t - t0) / 0.8) * (1 - smooth((t - 10) / 1.0)))

            pfade = smooth((t - 11) / 2.5)
            if pfade > 0:
                H = flow.hist
                segs = np.stack([H[:-1], H[1:]], axis=2).reshape(-1, 2, 2)
                d = np.hypot(*(H[1:] - H[:-1]).transpose(2, 0, 1))
                c = CMAP(NORM(flow.sst_at()))                     # (N,4)
                cols = np.repeat(c[None], TRAIL - 1, axis=0)       # (L-1,N,4)
                a = ramp[:, None] * 0.95 * pfade
                a = np.where(d > jump, 0, a)
                cols[..., 3] = a
                lc.set_segments(segs)
                lc.set_color(cols.reshape(-1, 4))
            else:
                lc.set_segments([])

            tfade = smooth((t - 14) / 1.5)
            title.set_alpha(tfade); sub.set_alpha(tfade)

            writer.grab_frame(facecolor=BG)
            if f % FPS == 0:
                print(f"{f}/{total}")
    print("Saved", OUT)

if __name__ == "__main__":
    main()