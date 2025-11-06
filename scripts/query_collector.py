from src.collector import Config, DuneCollector

if __name__ == "__main__":
    config = Config()
    collector = DuneCollector(
        config=config,
        max_workers=20,
        delay=0.1,
        retry_config={"max_retries": 100, "backoff_factor": 2.0, "retry_on_statuses": (429, 500, 502, 503, 504)}
    )
    collector.collect_queries(start_id=62001, end_id=200000, batch_size=2_000)