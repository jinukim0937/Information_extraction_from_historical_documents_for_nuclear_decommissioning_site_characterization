from ContaminationExtractor import Paper_Contamination
import os
import sys
import time

if __name__ == '__main__':
    arg1 = sys.argv[1]
    # appdata_path = os.getenv('APPDATA')

    if arg1 == "ExtractDocuments":
        paper_path = sys.argv[2]  # 'C:\\Users\\user\\Desktop\\input\\'
        save_path = sys.argv[3]  # 'C:\\Users\\user\\Desktop\\output\\'
        # paper_name = sys.argv[4]  # 'report.txt'
        historical_data = 'C:\\Users\\user\\Desktop\\input\\' + 'historical_data.xlsx'
        # Paper_Contamination(save_path, paper_path, historical_data)

    print("Please Wait 3 seconds...")
    time.sleep(3)
    # print("wake up!")

