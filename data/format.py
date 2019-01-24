from config import *
import csv

def export_predictions(predictions):
	with open(PREDICTIONS_CSV, mode='w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',')
		header = ['Id', 'rating']

		csv_writer.writerow(header)

		for prediction in predictions:
			row = [prediction['id'], prediction['rating']]
			csv_writer.writerow(row)
