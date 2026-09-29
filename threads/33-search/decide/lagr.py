"""per-layer Lagrangian allocation of churn: choose arm per layer (within a family) maximizing sal - mu*churn on
calib-val, apply same per-layer choice to heldout; sweep mu -> curve; interp at churn 3.2."""
import sys, numpy as np
import dlib as D
f = sys.argv[1] if len(sys.argv) > 1 else f"{D.W}/arms_v2.npz"
z = np.load(f); arms = list(z["arms"]); C, H = z["calib_fit"], z["glm52_heldout"]
fams = sorted({a.split(":")[0] for a in arms})
def curve(idx, A, B):
    pts = []
    for mu in np.geomspace(1e-5, 3e-2, 60):
        pick = np.argmax(A[:, idx, 0] - mu * A[:, idx, 1], 1)
        g = lambda M: (M[np.arange(len(pick)), np.array(idx)[pick], 0].mean(), M[np.arange(len(pick)), np.array(idx)[pick], 1].mean())
        pts.append((mu, *g(A), *g(B)))
    return np.array(pts)
for fam in fams + ["ALL"]:
    idx = [i for i, a in enumerate(arms) if fam == "ALL" or a.startswith(fam + ":")]
    glob_c = D.interp_at(C[:, idx, 0].mean(0), C[:, idx, 1].mean(0)); glob_h = D.interp_at(H[:, idx, 0].mean(0), H[:, idx, 1].mean(0))
    P = curve(idx, C, H)
    print(f"{fam:6s} global@3.2 calib {glob_c*100:.2f} held {glob_h*100:.2f} | per-layer@3.2 calib {D.interp_at(P[:,1],P[:,2])*100:.2f} held {D.interp_at(P[:,3],P[:,4])*100:.2f}")
