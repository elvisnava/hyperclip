import os

import wandb

base_path = os.path.dirname(os.path.dirname(__file__))

def populate_wandb_table(wandb_val_table, test_data, test_tasks, val_task_id, y_pred, y_true):
    ans = test_data[test_tasks[val_task_id]]["answers"]
    for i in range(len(y_pred)):
        image_name = test_data[test_tasks[val_task_id]]["test"][i][0]
        image_folder = image_name.split("_")[1]
        wandb_val_table.add_data(
                        f"{val_task_id}_{i}",
                        test_tasks[val_task_id],
                        wandb.Image(base_path + f'/data/VQA/Images/{image_folder}/{image_name}.jpg'),
                        ans[y_pred[i]],
                        ans[y_true[i]],
                        int(y_true[i]==y_pred[i]))
