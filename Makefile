
load_data:
	# Target to download data files
	@echo "⏳ Downloading data..."
	@python -m src.data.load_data
	@echo "✅ Data downloaded."
process_data:
	# Target to process data files
	@echo "⏳ Processing data..."
	@python -m src.data.process_data
	@echo "✅ Data processed."
split_and_store_data:
	# Target to ingest clean data files
	@echo "⏳ Splitting and storing data in vector store..."
	@python -m src.data.split_and_store_data --clear_db
	@echo "✅ Data split and stored."