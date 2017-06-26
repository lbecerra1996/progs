import yaml

with open("calibration.yaml") as f:
	data = yaml.load(f)

print data