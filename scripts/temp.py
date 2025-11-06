from src.matching.run_matching import main as run_matching
from src.evaluation.evaluate_results import main as evaluate

print("\n[5/6] 🤝 Matching e fusione punteggi...")
run_matching(test_mode=True)

print("\n[6/6] 📊 Valutazione risultati...")
evaluate()
