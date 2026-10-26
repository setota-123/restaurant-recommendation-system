#!/usr/bin/env python3
"""
Restaurant Recommendation System with Geospatial Analysis (Zomato dataset)
Author: Setota Gizaw - Jimma University

Usage:
    python restaurant_recommender.py            (CSV is auto-detected)
    python restaurant_recommender.py --data zomato.csv
    python restaurant_recommender.py --data zomato.csv --query "Sushi Loko" --city "Brasilia"

Outputs (in ./output): 3 charts (PNG), results.json, Restaurant_Recommendation_Report.pdf
Install:  pip install pandas numpy scikit-learn matplotlib reportlab
"""
import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

COLS = ["Restaurant ID", "Restaurant Name", "Country Code", "City", "Address",
        "Locality", "Locality Verbose", "Longitude", "Latitude", "Cuisines"]
COUNTRIES = {1: "India", 14: "Australia", 30: "Brazil", 37: "Canada", 94: "Indonesia",
             148: "New Zealand", 162: "Philippines", 166: "Qatar", 184: "Singapore",
             189: "South Africa", 191: "Sri Lanka", 208: "Turkey", 214: "UAE",
             215: "United Kingdom", 216: "United States"}
DECAY_KM = 5.0   # distance scale for the proximity score
EARTH_KM = 6371.0


# ----------------------------------------------------------------- 1. DATA
def find_csv():
    """Looks for the Zomato CSV in Kaggle, Colab and the current folder."""
    import glob
    for root in ("/kaggle/input", "/content", "."):
        for f in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
            if "sample_data" in f or "zomato_std" in f:
                continue
            try:
                head = pd.read_csv(f, nrows=2, encoding="latin-1")
            except Exception:
                continue
            if any("cuisine" in c.lower() for c in head.columns):
                return f
    raise FileNotFoundError("Zomato CSV not found. Add the dataset or pass --data path/to/file.csv")


def load_data(path):
    """Reads the Zomato CSV (utf-8 or latin-1) and maps its columns to the 10 names used here."""
    raw = None
    for enc in ("utf-8", "latin-1"):
        try:
            raw = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise ValueError("Could not read the file")
    if not any("cuisine" in str(c).lower() for c in raw.columns):      # header-less file
        raw = pd.read_csv(path, encoding="latin-1", header=None, usecols=range(10), names=COLS)

    def pick(*keys, exclude=()):
        for k in keys:
            for c in raw.columns:
                cl = str(c).lower()
                if k in cl and not any(e in cl for e in exclude):
                    return c
        return None

    m = {"Restaurant ID": pick("restaurant id", "id"), "Restaurant Name": pick("restaurant name", "name"),
         "Country Code": pick("country"), "City": pick("city"), "Address": pick("address"),
         "Locality": pick("locality", exclude=("verbose",)), "Locality Verbose": pick("locality verbose"),
         "Longitude": pick("longitude", "lon"), "Latitude": pick("latitude", "lat"), "Cuisines": pick("cuisine")}
    for need in ("Restaurant Name", "Longitude", "Latitude", "Cuisines"):
        if m[need] is None:
            raise ValueError(f"Column for '{need}' not found. Columns are: {list(raw.columns)}")
    df = pd.DataFrame({k: (raw[v] if v else None) for k, v in m.items()})
    if m["Restaurant ID"] is None: df["Restaurant ID"] = range(len(df))
    if m["Country Code"] is None:  df["Country Code"] = 0
    if m["City"] is None:          df["City"] = "Unknown"
    if m["Address"] is None:       df["Address"] = ""
    if m["Locality"] is None:      df["Locality"] = df["City"]
    if m["Locality Verbose"] is None: df["Locality Verbose"] = df["Locality"]
    return df


def clean_data(df):
    df = df[COLS].copy()
    n0 = len(df)
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df = df.dropna(subset=["Restaurant Name", "Cuisines", "Longitude", "Latitude", "City"])
    df = df[(df["Longitude"] != 0) | (df["Latitude"] != 0)]
    df = df[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]
    df = df.drop_duplicates(subset="Restaurant ID")
    for c in ["Restaurant Name", "City", "Locality", "Cuisines"]:
        df[c] = df[c].astype(str).str.strip()
    df["Country"] = df["Country Code"].map(COUNTRIES).fillna("Other")
    return df.reset_index(drop=True), n0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------- 2. RECOMMENDER
class RestaurantRecommender:
    """Hybrid content-based recommender: cuisine similarity + geographic proximity."""

    def __init__(self, df, alpha=0.7):
        self.df = df.reset_index(drop=True)
        self.alpha = alpha
        self.tokens = [set(c.strip().lower() for c in s.split(",") if c.strip()) for s in self.df["Cuisines"]]
        self.vec = TfidfVectorizer(tokenizer=lambda s: [c.strip().lower() for c in s.split(",") if c.strip()],
                                   lowercase=False, token_pattern=None)
        self.X = self.vec.fit_transform(self.df["Cuisines"])
        self.lat = self.df["Latitude"].values
        self.lon = self.df["Longitude"].values

    def _find(self, name, city=None):
        m = self.df["Restaurant Name"].str.lower().str.contains(name.lower(), regex=False)
        if city:
            m &= self.df["City"].str.lower() == city.lower()
        hits = np.where(m)[0]
        if len(hits) == 0:
            raise ValueError(f"Restaurant '{name}' not found")
        return hits[0]

    def recommend_similar(self, name, city=None, top_n=5, max_km=None):
        """Restaurants similar to `name` (same cuisines) and close to it."""
        i = self._find(name, city)
        same_city = (self.df["City"] == self.df.at[i, "City"]).values
        cand = np.where(same_city & (np.arange(len(self.df)) != i))[0]
        sims = cosine_similarity(self.X[i], self.X[cand]).ravel()
        dist = haversine_km(self.lat[i], self.lon[i], self.lat[cand], self.lon[cand])
        score = self.alpha * sims + (1 - self.alpha) * np.exp(-dist / DECAY_KM)
        out = self.df.iloc[cand][["Restaurant Name", "Locality", "Cuisines"]].copy()
        out["Similarity"], out["Distance_km"], out["Score"] = sims, dist, score
        if max_km:
            out = out[out["Distance_km"] <= max_km]
        return out.sort_values("Score", ascending=False).head(top_n).round(3).reset_index(drop=True)

    def recommend_nearby(self, lat, lon, cuisine, top_n=5, radius_km=5):
        """Restaurants of a wanted cuisine around a GPS point (user location)."""
        q = self.vec.transform([cuisine])
        sims = cosine_similarity(q, self.X).ravel()
        dist = haversine_km(lat, lon, self.lat, self.lon)
        score = self.alpha * sims + (1 - self.alpha) * np.exp(-dist / DECAY_KM)
        out = self.df[["Restaurant Name", "Locality", "Cuisines"]].copy()
        out["Similarity"], out["Distance_km"], out["Score"] = sims, dist, score
        out = out[(out["Distance_km"] <= radius_km) & (out["Similarity"] > 0)]
        return out.sort_values("Score", ascending=False).head(top_n).round(3).reset_index(drop=True)


# ----------------------------------------------------------- 3. EVALUATION
def evaluate(rec, k=5, n_queries=500, seed=42):
    """Precision@k = share of top-k recommendations that share >=1 cuisine with the query."""
    rng = np.random.default_rng(seed)
    df = rec.df
    idxs = rng.choice(len(df), size=min(n_queries, len(df)), replace=False)
    p = {"hybrid": [], "content": [], "random": []}
    d = {"hybrid": [], "content": [], "random": []}
    jac = []
    for i in idxs:
        cand = np.where((df["City"].values == df.at[i, "City"]) & (np.arange(len(df)) != i))[0]
        if len(cand) < k:
            continue
        sims = cosine_similarity(rec.X[i], rec.X[cand]).ravel()
        dist = haversine_km(rec.lat[i], rec.lon[i], rec.lat[cand], rec.lon[cand])
        hybrid = rec.alpha * sims + (1 - rec.alpha) * np.exp(-dist / DECAY_KM)
        orders = {"hybrid": np.argsort(-hybrid, kind="stable"),
                  "content": np.argsort(-sims, kind="stable"),
                  "random": rng.permutation(len(cand))}
        q = rec.tokens[i]
        for name, o in orders.items():
            top = o[:k]
            p[name].append(np.mean([len(q & rec.tokens[cand[j]]) > 0 for j in top]))
            d[name].append(dist[top].mean())
            if name == "hybrid":
                jac.append(np.mean([len(q & rec.tokens[cand[j]]) / len(q | rec.tokens[cand[j]]) for j in top]))
    return {"k": k, "queries_evaluated": len(p["hybrid"]),
            "precision": {m: round(float(np.mean(v)), 3) for m, v in p.items()},
            "avg_distance_km": {m: round(float(np.mean(v)), 2) for m, v in d.items()},
            "mean_jaccard_hybrid": round(float(np.mean(jac)), 3)}


# ------------------------------------------------------ 4. GEOSPATIAL / EDA
def hotspots(df, eps_km=0.5, min_samples=None):
    """DBSCAN with haversine metric finds dense restaurant clusters (food hotspots)."""
    ms = min_samples or (5 if len(df) > 500 else 3)
    coords = np.radians(df[["Latitude", "Longitude"]].values)
    labels = DBSCAN(eps=eps_km / EARTH_KM, min_samples=ms, metric="haversine",
                    algorithm="ball_tree").fit_predict(coords)
    df = df.copy()
    df["Cluster"] = labels
    g = df[df["Cluster"] >= 0].groupby("Cluster")
    top = []
    for cid, grp in sorted(g, key=lambda x: -len(x[1]))[:5]:
        top.append({"area": grp["Locality"].mode().iloc[0], "city": grp["City"].mode().iloc[0],
                    "restaurants": int(len(grp)),
                    "top_cuisine": grp["Cuisines"].str.split(",").explode().str.strip().value_counts().index[0]})
    return df, {"clusters": int(labels.max() + 1) if labels.max() >= 0 else 0,
                "noise_points": int((labels == -1).sum()), "eps_km": eps_km, "min_samples": ms, "top": top}


def make_charts(df, outdir):
    paths = {}
    cu = df["Cuisines"].str.split(",").explode().str.strip().value_counts().head(10)[::-1]
    plt.figure(figsize=(7, 4)); cu.plot(kind="barh", color="#2E86AB")
    plt.title("Top 10 cuisines"); plt.xlabel("Number of restaurants"); plt.tight_layout()
    paths["cuisines"] = os.path.join(outdir, "top_cuisines.png"); plt.savefig(paths["cuisines"], dpi=150); plt.close()

    ci = df["City"].value_counts().head(10)[::-1]
    plt.figure(figsize=(7, 4)); ci.plot(kind="barh", color="#F18F01")
    plt.title("Restaurants per city (top 10)"); plt.xlabel("Number of restaurants"); plt.tight_layout()
    paths["cities"] = os.path.join(outdir, "restaurants_per_city.png"); plt.savefig(paths["cities"], dpi=150); plt.close()

    big = df["City"].value_counts().index[0]
    sub = df[df["City"] == big]
    plt.figure(figsize=(7, 5))
    noise = sub[sub["Cluster"] == -1]
    plt.scatter(noise["Longitude"], noise["Latitude"], c="lightgray", s=18, label="Isolated")
    cl = sub[sub["Cluster"] >= 0]
    if len(cl):
        plt.scatter(cl["Longitude"], cl["Latitude"], c=cl["Cluster"], cmap="tab10", s=30, label="Hotspot clusters")
    plt.title(f"Geospatial hotspots (DBSCAN) - {big}"); plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.legend(); plt.tight_layout()
    paths["map"] = os.path.join(outdir, "hotspot_map.png"); plt.savefig(paths["map"], dpi=150); plt.close()
    paths["map_city"] = big
    return paths


# --------------------------------------------------------------- 5. REPORT
def build_pdf(path, stats, ev, hs, demos, charts, author, school):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10.5, leading=15, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor("#1B3A5C"), spaceBefore=10)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=22, leading=27, textColor=colors.HexColor("#1B3A5C"))
    small = ParagraphStyle("s", parent=body, fontSize=8.5, leading=11)
    cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10, spaceAfter=0)

    def tbl(data, widths, header=True):
        rows = [[Paragraph(str(x), cell) for x in r] for r in data]
        t = Table(rows, colWidths=widths, repeatRows=1)
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE8F5")))
        t.setStyle(TableStyle(st))
        return t

    P = lambda t, s=body: Paragraph(t, s)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
                            title="Restaurant Recommendation System", author=author)
    s = []
    s += [P("Restaurant Recommendation System with Geospatial Analysis", title),
          P(f"<b>Author:</b> {author} &nbsp;|&nbsp; {school} &nbsp;|&nbsp; {datetime.date.today():%B %d, %Y}",
            ParagraphStyle("a", parent=body, alignment=1)), Spacer(1, 10)]

    s += [P("Abstract", h1),
          P(f"This project builds an AI-based restaurant recommendation system on the Zomato restaurants dataset. "
            f"The system combines <b>content-based filtering</b> (TF-IDF vectors of cuisines compared with cosine similarity) "
            f"with <b>geospatial analysis</b> (Haversine distance and DBSCAN density clustering) to recommend restaurants that "
            f"are both similar in food type and close to the user. On {stats['rows']:,} cleaned restaurants from "
            f"{stats['cities']} cities, the hybrid model reached a Precision@{ev['k']} of {ev['precision']['hybrid']:.2f} "
            f"against {ev['precision']['random']:.2f} for random recommendations, while keeping recommendations about "
            f"{ev['avg_distance_km']['hybrid']} km away on average (random: {ev['avg_distance_km']['random']} km).")]

    s += [P("1. Problem Statement", h1),
          P("People in a new city face thousands of restaurants and little personalized guidance. Existing lists rank "
            "places by popularity only and ignore where the user is. The goal is a recommender that answers two questions: "
            "<i>(a) which restaurants are similar to the one I like?</i> and <i>(b) which of them are near me?</i>")]

    s += [P("2. Dataset", h1),
          P(f"The dataset is the Zomato restaurants data (Kaggle). Each row is one restaurant with its ID, name, country code, "
            f"city, address, locality, longitude, latitude and a comma-separated list of cuisines. After cleaning "
            f"(removing {stats['removed']} rows with missing cuisines/coordinates or invalid (0,0) coordinates, and duplicates) "
            f"{stats['rows']:,} restaurants in {stats['countries']} countries and {stats['cities']} cities remain, "
            f"with {stats['unique_cuisines']} distinct cuisines.")]
    s += [tbl([["Column", "Type", "Use in the project"],
               ["Restaurant ID / Name", "ID / text", "Identification and display"],
               ["Country Code, City, Locality", "categorical", "Filtering by place; hotspot naming"],
               ["Longitude, Latitude", "numeric", "Haversine distance, DBSCAN clustering, maps"],
               ["Cuisines", "text (multi-label)", "TF-IDF features for similarity"]],
              [5*cm, 3.5*cm, 8.5*cm]), Spacer(1, 6)]

    s += [P("3. Methodology", h1),
          P("<b>3.1 Preprocessing.</b> Files are read with UTF-8 or Latin-1 encoding, coordinates are converted to numbers, "
            "invalid rows are removed, and text is trimmed."),
          P("<b>3.2 Content-based filtering.</b> Each restaurant's cuisine list is converted to a TF-IDF vector. Rare cuisines "
            "(e.g. Peruvian) receive higher weight than common ones (e.g. Cafe). Similarity between two restaurants is the cosine "
            "of their vectors."),
          P("<b>3.3 Geospatial proximity.</b> The great-circle distance <i>d</i> (km) between two restaurants is computed "
            "with the Haversine formula and converted to a proximity score <i>exp(-d / 5)</i>, which is 1 for the same spot "
            "and decays smoothly with distance."),
          P("<b>3.4 Hybrid score.</b> <i>Score = 0.7 x cosine_similarity + 0.3 x proximity</i>. Candidates are restricted to "
            "the same city and ranked by score. A second function recommends a wanted cuisine around the user's GPS point "
            "within a chosen radius."),
          P(f"<b>3.5 Hotspot detection.</b> DBSCAN with the Haversine metric (eps = {hs['eps_km']} km, min_samples = "
            f"{hs['min_samples']}) groups restaurants into dense food districts without choosing the number of clusters."),
          P("<b>3.6 Evaluation.</b> For randomly chosen restaurants, the top-5 recommendations from the same city are checked: "
            "a recommendation counts as relevant if it shares at least one cuisine with the query. Three strategies are compared: "
            "hybrid, content-only and random.")]

    s += [PageBreak(), P("4. Results", h1), P("<b>4.1 Exploratory analysis</b>")]
    s += [Image(charts["cuisines"], width=11*cm, height=6.3*cm), Image(charts["cities"], width=11*cm, height=6.3*cm)]

    s += [P("<b>4.2 Recommendation quality</b>"),
          tbl([["Strategy", f"Precision@{ev['k']}", "Avg. distance of recommendations (km)"],
               ["Hybrid (cuisine + distance)", ev["precision"]["hybrid"], ev["avg_distance_km"]["hybrid"]],
               ["Content-only (cuisine)", ev["precision"]["content"], ev["avg_distance_km"]["content"]],
               ["Random baseline", ev["precision"]["random"], ev["avg_distance_km"]["random"]]],
              [6*cm, 4*cm, 7*cm]),
          P(f"Evaluated on {ev['queries_evaluated']} query restaurants. Mean Jaccard overlap of the hybrid recommendations: "
            f"{ev['mean_jaccard_hybrid']}. The hybrid model keeps almost all of the cuisine accuracy of the content-only model "
            f"while recommending closer places, and is far better than random.", small), Spacer(1, 6)]

    s += [P("<b>4.3 Example recommendations</b>")]
    for d in demos:
        s += [P(f"Because you like <b>{d['query']}</b> ({d['city']}; {d['cuisines']}):", body)]
        rows = [["Recommended restaurant", "Area", "Cuisines", "Sim.", "Km", "Score"]]
        for _, r in d["table"].iterrows():
            rows.append([r["Restaurant Name"], r["Locality"], r["Cuisines"], r["Similarity"], r["Distance_km"], r["Score"]])
        s += [tbl(rows, [4.2*cm, 3.3*cm, 4.3*cm, 1.4*cm, 1.4*cm, 1.4*cm]), Spacer(1, 6)]

    s += [PageBreak(), P("4.4 Geospatial hotspots", ParagraphStyle("h2", parent=h1, fontSize=12)),
          P(f"DBSCAN found <b>{hs['clusters']}</b> dense restaurant clusters; {hs['noise_points']} restaurants are isolated. "
            f"The map below shows {charts['map_city']}."),
          Image(charts["map"], width=12.5*cm, height=8.9*cm)]
    if hs["top"]:
        rows = [["Hotspot area", "City", "Restaurants", "Most common cuisine"]]
        rows += [[h["area"], h["city"], h["restaurants"], h["top_cuisine"]] for h in hs["top"]]
        s += [tbl(rows, [6*cm, 4*cm, 3*cm, 4*cm])]

    s += [P("5. Implementation", h1),
          P("The system is written in Python 3 with pandas and NumPy (data), scikit-learn (TF-IDF, cosine similarity, DBSCAN), "
            "Matplotlib (charts) and ReportLab (this report). The class <i>RestaurantRecommender</i> exposes "
            "<i>recommend_similar(name, city)</i> and <i>recommend_nearby(lat, lon, cuisine, radius_km)</i>. "
            "One command runs the full pipeline: cleaning, analysis, evaluation, charts and this PDF."),
          P("6. Conclusion and Future Work", h1),
          P("The project shows that combining text features (cuisines) with geographic information gives recommendations that "
            "are relevant and nearby, without needing user history (no cold-start problem). Future work: add ratings, price range "
            "and reviews to the score; use collaborative filtering when user data exists; deploy as a web or mobile app with a "
            "live map; and evaluate with real user feedback."),
          P("7. References", h1),
          P("Zomato Restaurants Data, Kaggle. | Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 2011. | "
            "Ester et al., A density-based algorithm for discovering clusters (DBSCAN), KDD 1996. | "
            "Sinnott, Virtues of the Haversine, Sky and Telescope 1984.", small),
          P(f"Source code: restaurant_recommender.py (GitHub: setota-123).", small)]
    doc.build(s)


# -------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to the Zomato CSV (auto-detected if omitted)")
    ap.add_argument("--out", default="/kaggle/working/output" if os.path.isdir("/kaggle/working") else "output")
    ap.add_argument("--query", help="restaurant name to get recommendations for")
    ap.add_argument("--city", help="city of the query restaurant")
    ap.add_argument("--author", default="Setota Gizaw")
    ap.add_argument("--school", default="Jimma University, Software Engineering")
    a = ap.parse_known_args()[0]      # ignores notebook kernel arguments
    os.makedirs(a.out, exist_ok=True)

    data_path = a.data or find_csv()
    print('Using data file:', data_path)
    df, n0 = clean_data(load_data(data_path))
    print(f"Loaded {n0} rows -> {len(df)} after cleaning")
    rec = RestaurantRecommender(df)

    if a.query:   # quick interactive use
        print(rec.recommend_similar(a.query, a.city).to_string(index=False))

    ev = evaluate(rec)
    df_h, hs = hotspots(df)
    charts = make_charts(df_h, a.out)

    # demo recommendations: one restaurant in each of the 3 biggest cities
    demos = []
    for city in df["City"].value_counts().index[:3]:
        cand = df[(df["City"] == city) & (df["Cuisines"].str.contains(","))]
        row = (cand if len(cand) else df[df["City"] == city]).iloc[0]
        demos.append({"query": row["Restaurant Name"], "city": city, "cuisines": row["Cuisines"],
                      "table": rec.recommend_similar(row["Restaurant Name"], city, top_n=3)})
    # nearby demo (user standing at the first restaurant)
    r0 = df.iloc[0]
    nearby = rec.recommend_nearby(r0["Latitude"], r0["Longitude"], r0["Cuisines"].split(",")[0], top_n=3, radius_km=3)

    stats = {"rows": len(df), "removed": n0 - len(df), "cities": df["City"].nunique(),
             "countries": df["Country"].nunique(),
             "unique_cuisines": df["Cuisines"].str.split(",").explode().str.strip().nunique()}
    print("Evaluation:", json.dumps(ev, indent=2))
    print("Hotspots:", json.dumps(hs, indent=2))
    for d in demos:
        print(f"\nBecause you like {d['query']} ({d['city']}):\n", d["table"].to_string(index=False))
    print("\nNearby demo:\n", nearby.to_string(index=False))

    json.dump({"stats": stats, "evaluation": ev, "hotspots": hs}, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    pdf = os.path.join(a.out, "Restaurant_Recommendation_Report.pdf")
    try:
        try:
            import reportlab  # noqa
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "reportlab"], check=False)
        build_pdf(pdf, stats, ev, hs, demos, charts, a.author, a.school)
        print("\nDone. Report:", pdf)
    except Exception as e:
        print("\nProject results are above. PDF skipped:", e)


if __name__ == "__main__":
    main()
