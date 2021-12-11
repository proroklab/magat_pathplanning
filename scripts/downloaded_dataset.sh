mkdir Data
cd Data
gdown https://drive.google.com/uc\?id\=1fyWp6tHenZlaimS6-f7hBwD-sqzNtcbz
gdown https://drive.google.com/uc\?id\=1Pj5EJKb0kIuY8xGnpq2PDx1qQMvd2mgq

unzip DataSource_DMap_FixedComR.zip
unzip trained_model.zip

mv trained_model/experiments experiments
rm DataSource_DMap_FixedComR.zip
rm trained_model.zip