import train_pipeline
import datetime

def start_retraining():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Retraining process initiated...")
    try:
        train_pipeline.train()
        print(f"[{timestamp}] Retraining successfully completed.")
    except Exception as e:
        print(f"[{timestamp}] Retraining Failed: {e}")

if __name__ == "__main__":
    start_retraining()
