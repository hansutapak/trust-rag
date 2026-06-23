from src.ingestion.static_loader import get_vectorstore, load_json_data


def auto_seed():
    vectorstore = get_vectorstore()

    # works only if the databasse is empty
    if vectorstore._collection.count() == 0:
        print("🔄 Database is empty, loading documents automatically...")
        load_json_data(max_per_file=50, batch_size=50, delay_seconds=60)
        print("✅ static documents loaded successfully.")
    else:
        print("Database already contains data, Skipping setup.")


if __name__ == "__main__":
    auto_seed()
