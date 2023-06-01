import yaml
import json
import re
import sys

def update_version(fiftyone_yml_file, package_json_file, new_version):
    # Read fiftyone.yml file
    with open(fiftyone_yml_file, 'r') as f:
        fiftyone_yml_data = yaml.safe_load(f)

    # Read package.json file
    with open(package_json_file, 'r') as f:
        package_json_data = json.load(f)

    # Get current versions
    fiftyone_version = fiftyone_yml_data['version']
    package_version = package_json_data['version']

    # Update versions based on the input
    if new_version == "bump":
        # Bump the last digit in the semver version string
        fiftyone_version = bump_version(fiftyone_version)
        package_version = bump_version(package_version)
    else:
        # Update versions with the exact provided version
        fiftyone_version = new_version
        package_version = new_version

    # Update the version in fiftyone.yml file
    fiftyone_yml_data['version'] = fiftyone_version

    # Update the version in package.json file
    package_json_data['version'] = package_version

    # Write updated data back to the files
    with open(fiftyone_yml_file, 'w') as f:
        yaml.dump(fiftyone_yml_data, f, default_flow_style=False)

    with open(package_json_file, 'w') as f:
        json.dump(package_json_data, f, indent=4)

    print('Versions updated successfully.', fiftyone_version)


def bump_version(version):
    # Regex pattern to match the last digit in the semver version string
    pattern = r'(\d+)(?!.*\d)'

    # Find the last digit and increment it by 1
    match = re.search(pattern, version)
    if match:
        last_digit = int(match.group(1))
        new_last_digit = last_digit + 1
        version = re.sub(pattern, str(new_last_digit), version)

    return version


# Example usage
fiftyone_yml_file = 'fiftyone.yml'
package_json_file = 'package.json'
new_version = sys.argv[1]  # Pass the new version as a command-line argument
explicit_version = None
try:
    explicit_version = sys.argv[2]
except:
    pass
update_version(fiftyone_yml_file, package_json_file, explicit_version or new_version)