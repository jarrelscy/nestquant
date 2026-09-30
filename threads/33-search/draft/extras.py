import dlib as D
D.register("last", ["last1_cnt", "last1_sal", "last2_cnt", "last2_sal", "last4_cnt", "last4_sal"])
D.register("draft", [f"d{k}_{s}" for k in (1, 2, 4) for s in ("cnt", "sal", "soft")] +
           [f"o{k}_{s}" for k in (1, 2, 4) for s in ("cnt", "sal")])
D.register("tok", [f"tk{k}_{s}" for k in (1, 2, 4) for s in ("cnt", "sal")])
