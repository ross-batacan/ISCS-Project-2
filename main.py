import logging
from etl import ETLProcess  # Import the ETLProcess class from etl.py

# Configure logging for the main entry point of the program
logging.basicConfig(
    filename='main_process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    try:
        logging.info("Starting the ETL process...")
        
        # Instantiate the ETLProcess class
        etl_process = ETLProcess()
        
        # Run the ETL process
        etl_process.run_etl()
        
        logging.info("ETL process completed successfully.")
        print("ETL process completed successfully.")
    
    except Exception as e:
        logging.error(f"An error occurred in the main process: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
