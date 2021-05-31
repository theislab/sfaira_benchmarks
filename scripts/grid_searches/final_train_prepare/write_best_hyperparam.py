import os
import sys

# Process sys arguments
model_class = str(sys.argv[1])
organism = str(sys.argv[2])
organs = str(sys.argv[3]).lower().split(' ')
gs_keys = str(sys.argv[4])
grid_search_dir = str(sys.argv[5])
metric = str(sys.argv[6])
out_path_base = str(sys.argv[7])

# Set directory path
final_training_dir = os.path.join(out_path_base, 'final_training', organism, model_class, gs_keys, 'hyperparameter')
grid_search_dir = os.path.join(grid_search_dir, organism, model_class)

# Initialize gridsearch container
mod = __import__(
    'sfaira.train',
    fromlist=[f'SummarizeGridsearch{model_class.capitalize()}']
)
SummarizeGS = getattr(
    mod,
    f'SummarizeGridsearch{model_class.capitalize()}'
)

cv = True
gs_keys = gs_keys.split("+")
summarize_gs = SummarizeGS(
    source_path=dict([(x, grid_search_dir) for x in gs_keys]),
    cv=cv)
summarize_gs.load_gs(gs_ids=gs_keys)
summarize_gs.create_summary_tab()

# Write best hyperparameter
for i, o in enumerate(organs):
    summarize_gs.write_best_hyparam(
        write_path=final_training_dir,
        subset={"organ": o},
        partition="test",
        metric=metric,
        cvs=[0] if cv else None  # holdout dataset irrelevant here
    )
