from django.shortcuts import render
from joblib import load
from sklearn.preprocessing import LabelEncoder
import pandas as pd

model = load('./SavedModels/model.joblib') 

label_encoder_toss_decision = LabelEncoder()
label_encoder_venue = LabelEncoder()

label_encoder_toss_decision.fit(['bat', 'field'])
label_encoder_venue.fit(['Home', 'Away'])

teams = ['Pune Warriors', 
         'Kolkata Knight Riders',
         'Rajasthan Royals',
         'Kochi Tuskers Kerala',
         'Gujarat Lions',
         'Chennai Super Kings', 
         'Delhi Daredevils', 
         'Deccan Chargers', 
         'Delhi Capitals',
         'Mumbai Indians', 
         'Sunrisers Hyderabad', 
         'Rising Pune Supergiants', 
         'Royal Challengers Bangalore', 
         'Kings XI Punjab']

def predictor(request):
    if request.method == 'POST':
        toss_winner = request.POST.get('toss_winner')
        toss_decision = request.POST.get('toss_decision')
        venue = request.POST.get('venue')

        toss_winner_encoded = pd.get_dummies(pd.Series(toss_winner)).reindex(columns=teams, fill_value=0)
    
        # Label encoding for toss decision and venue
        toss_decision_encoded = label_encoder_toss_decision.transform([toss_decision])[0]
        venue_encoded = label_encoder_venue.transform([venue])[0]
        
        # Combine encoded features into a single array
        input_features = list(toss_winner_encoded.values.flatten()) + [toss_decision_encoded, venue_encoded]

        y_pred = model.predict([input_features])
        if y_pred[0] == 0:
            y_pred = 'lose'
        else:
            y_pred = 'win'
        return render(request, 'main.html', {'result' : y_pred})
    return render(request, 'main.html')

    
