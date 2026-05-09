from synthe import PCARVFLSimulator, adequacy_report
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

datasets = {
    "iris":          load_iris(return_X_y=True)[0],
    "wine":          load_wine(return_X_y=True)[0],
    "breast_cancer": load_breast_cancer(return_X_y=True)[0],
    "digits":        load_digits(return_X_y=True)[0],
}

for name, X in datasets.items():
    print(f"\n{'=' * 56}")
    print(f"  {name.upper()}  (n={X.shape[0]}, d={X.shape[1]})")
    print(f"{'=' * 56}")
    sim   = PCARVFLSimulator(random_state=42)
    sim.fit(X, n_trials=30)
    X_syn = sim.sample(400)
    adequacy_report(X, X_syn)