import argparse

import requests

########################################################################################

from quantumion.server.model import UserRegistrationForm

########################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        default="http://127.0.0.1:8000",
        type=str,
        help="server URL for job submission",
    )

    args = parser.parse_args()

    BASE_URL = args.url

    username = "string"
    email = "string"
    password = "string"

    registration = UserRegistrationForm(
        username=username, email=email, password=password
    )

    registration_url = BASE_URL + "/user/register"
    response = requests.post(registration_url, json=registration.model_dump())
    print(response.json())

    response.raise_for_status()
