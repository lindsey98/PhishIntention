import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pprint
import pandas as pd
from gspread.models import Cell


class gwrapper():
    def __init__(self):
        scope = [
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/drive.file'
        ]
        file_name = 'cred.json'
        creds = ServiceAccountCredentials.from_json_keyfile_name(file_name, scope)
        client = gspread.authorize(creds)

        # Fetch the sheet
        self.sheet = client.open('test').sheet1

    def get_records(self):
        return self.sheet.get_all_records()

    def update_file(self, filename, date):
        df = pd.read_csv(filename, delimiter='\t')
        print(df.values)
        df['input_score'] = 'ERROR'
        df['yes'] = 0
        df['no'] = 0
        df['unsure'] = 0
        df = df[df['phish'] == 1]
        df = df.drop_duplicates('url', )
        df['date'] = date
        df = df[['date', 'url', 'foldername', 'prediction', 'input_score', 'vt_result', 'yes', 'no', 'unsure']] # columns here
        list2 = df.values.tolist()[1:]
        print(list2)
        results = self.sheet.append_rows(list2)
        print(results)

    def update_list(self, to_update):

        self.sheet.append_rows(to_update)

    def update_cell(self, row, yes, no, unsure):
        # update google sheet yes, no, unsure columns
        cells = []
        row = int(row)
        cells.append(Cell(row=row, col=7, value=yes))
        cells.append(Cell(row=row, col=8, value=no))
        cells.append(Cell(row=row, col=9, value=unsure))
        self.sheet.update_cells(cells)


if __name__ == '__main__':
    g = gwrapper()
    g.update_file('new_results/280820_xgpc.txt', '280220')
