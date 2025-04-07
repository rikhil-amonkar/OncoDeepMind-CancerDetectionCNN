import optuna

# Load model
from model.model import DrugResponseModel, criterion, X_train, x_scaler, y_scaler, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
import torch
import pandas as pd
from sklearn.metrics import r2_score

model = torch.load('backend/model/DrugResponseModel.pth')

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hl1 = trial.suggest_int('hl1', 64, 512)
    hl2 = trial.suggest_int('hl2', 32, 256)
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Initialize model with trial params
    model = DrugResponseModel(input_features=X_train.shape[1], 
                             hl1=hl1, hl2=hl2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop (simplified)
    for epoch in range(50):  # Fewer epochs for speed
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        r2 = r2_score(y_test_tensor.numpy(), y_pred.numpy())
    
    return r2  # Optuna will maximize this

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print(f"R²: {trial.value:.4f}")
print("Best params:")
for key, value in trial.params.items():
    print(f"  {key}: {value}")

#****************** STUDY RESULTS ******************

# [I 2025-04-07 11:27:59,553] A new study created in memory with name: no-name-c563efef-c5a8-497c-81aa-b663cc7e3f5e
# [I 2025-04-07 11:28:37,577] Trial 0 finished with value: 0.6306518316268921 and parameters: {'lr': 0.0006284380694799547, 'hl1': 507, 'hl2': 135, 'dropout': 0.2164859477464726}. Best is trial 0 with value: 0.6306518316268921.
# [I 2025-04-07 11:28:52,763] Trial 1 finished with value: 0.6527959108352661 and parameters: {'lr': 0.006860477184566008, 'hl1': 136, 'hl2': 114, 'dropout': 0.41956726822247176}. Best is trial 1 with value: 0.6527959108352661.
# [I 2025-04-07 11:29:24,471] Trial 2 finished with value: 0.08015561103820801 and parameters: {'lr': 1.8208760730674115e-05, 'hl1': 345, 'hl2': 199, 'dropout': 0.4219004084524862}. Best is trial 1 with value: 0.6527959108352661.
# [I 2025-04-07 11:29:58,803] Trial 3 finished with value: 0.013105452060699463 and parameters: {'lr': 1.0681747561695308e-05, 'hl1': 377, 'hl2': 238, 'dropout': 0.10464896835923013}. Best is trial 1 with value: 0.6527959108352661.
# [I 2025-04-07 11:30:28,566] Trial 4 finished with value: 0.6545093059539795 and parameters: {'lr': 0.0011712551115123453, 'hl1': 275, 'hl2': 256, 'dropout': 0.45534857177256993}. Best is trial 4 with value: 0.6545093059539795.
# [I 2025-04-07 11:30:43,855] Trial 5 finished with value: 0.5930216312408447 and parameters: {'lr': 0.00037186359951539425, 'hl1': 147, 'hl2': 106, 'dropout': 0.17099493221891598}. Best is trial 4 with value: 0.6545093059539795.
# [I 2025-04-07 11:30:56,497] Trial 6 finished with value: 0.609366774559021 and parameters: {'lr': 0.0006087091212875191, 'hl1': 110, 'hl2': 79, 'dropout': 0.35377818982465153}. Best is trial 4 with value: 0.6545093059539795.
# [I 2025-04-07 11:31:17,286] Trial 7 finished with value: 0.6510359644889832 and parameters: {'lr': 0.005752135188909479, 'hl1': 249, 'hl2': 120, 'dropout': 0.3973675786419051}. Best is trial 4 with value: 0.6545093059539795.
# [I 2025-04-07 11:31:47,967] Trial 8 finished with value: 0.6554012894630432 and parameters: {'lr': 0.0012849813609933643, 'hl1': 327, 'hl2': 223, 'dropout': 0.11486073874063672}. Best is trial 8 with value: 0.6554012894630432.
# [I 2025-04-07 11:32:08,735] Trial 9 finished with value: -0.03307211399078369 and parameters: {'lr': 1.1115447150219122e-05, 'hl1': 182, 'hl2': 201, 'dropout': 0.12625985769289458}. Best is trial 8 with value: 0.6554012894630432.
# [I 2025-04-07 11:32:44,459] Trial 10 finished with value: 0.39840734004974365 and parameters: {'lr': 7.79810218973599e-05, 'hl1': 458, 'hl2': 173, 'dropout': 0.2644990965524985}. Best is trial 8 with value: 0.6554012894630432.
# [I 2025-04-07 11:33:13,597] Trial 11 finished with value: 0.6609945893287659 and parameters: {'lr': 0.001763901395596742, 'hl1': 260, 'hl2': 256, 'dropout': 0.48582084114931706}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:33:45,732] Trial 12 finished with value: 0.6540089845657349 and parameters: {'lr': 0.001852663274465509, 'hl1': 364, 'hl2': 219, 'dropout': 0.49542009562464123}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:34:02,877] Trial 13 finished with value: 0.4793784022331238 and parameters: {'lr': 0.000141501301736657, 'hl1': 237, 'hl2': 34, 'dropout': 0.31096899658910765}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:34:30,232] Trial 14 finished with value: 0.6571191549301147 and parameters: {'lr': 0.002868439629018329, 'hl1': 306, 'hl2': 170, 'dropout': 0.2646466181337313}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:34:51,029] Trial 15 finished with value: 0.6413991451263428 and parameters: {'lr': 0.00314193665441923, 'hl1': 216, 'hl2': 166, 'dropout': 0.2640073836194371}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:35:24,665] Trial 16 finished with value: 0.641960620880127 and parameters: {'lr': 0.003250510430421725, 'hl1': 424, 'hl2': 170, 'dropout': 0.3400142614609688}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:35:45,263] Trial 17 finished with value: 0.6542384624481201 and parameters: {'lr': 0.0092157582085974, 'hl1': 274, 'hl2': 52, 'dropout': 0.22291432405467146}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:36:11,318] Trial 18 finished with value: 0.5819618701934814 and parameters: {'lr': 0.0001985781478754813, 'hl1': 306, 'hl2': 146, 'dropout': 0.36529423886260787}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:36:28,656] Trial 19 finished with value: 0.6609374284744263 and parameters: {'lr': 0.0037878012798238095, 'hl1': 65, 'hl2': 256, 'dropout': 0.49636014743232215}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:36:45,324] Trial 20 finished with value: 0.09184026718139648 and parameters: {'lr': 5.364623229103481e-05, 'hl1': 72, 'hl2': 251, 'dropout': 0.49330160006894097}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:37:06,526] Trial 21 finished with value: 0.6595228910446167 and parameters: {'lr': 0.0026095222211008997, 'hl1': 191, 'hl2': 205, 'dropout': 0.4656169752257314}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:37:29,593] Trial 22 finished with value: 0.6601195335388184 and parameters: {'lr': 0.004483998236615964, 'hl1': 205, 'hl2': 230, 'dropout': 0.4583661678465598}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:37:45,633] Trial 23 finished with value: 0.6571066379547119 and parameters: {'lr': 0.004595045561614288, 'hl1': 69, 'hl2': 234, 'dropout': 0.45377937526928863}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:38:07,440] Trial 24 finished with value: 0.649175763130188 and parameters: {'lr': 0.0010888453131499597, 'hl1': 173, 'hl2': 244, 'dropout': 0.49998823704294104}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:38:25,314] Trial 25 finished with value: 0.657718300819397 and parameters: {'lr': 0.0019058546389965603, 'hl1': 110, 'hl2': 229, 'dropout': 0.3967053045714377}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:38:48,013] Trial 26 finished with value: 0.6575777530670166 and parameters: {'lr': 0.009378385790232218, 'hl1': 215, 'hl2': 214, 'dropout': 0.4415753850264652}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:39:04,648] Trial 27 finished with value: 0.6109221577644348 and parameters: {'lr': 0.0006026993259080969, 'hl1': 109, 'hl2': 189, 'dropout': 0.46656958356656597}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:39:26,668] Trial 28 finished with value: 0.6509888172149658 and parameters: {'lr': 0.004445316526339423, 'hl1': 155, 'hl2': 252, 'dropout': 0.3803940844088178}. Best is trial 11 with value: 0.6609945893287659.
# [I 2025-04-07 11:39:51,150] Trial 29 finished with value: 0.6453964710235596 and parameters: {'lr': 0.0009551004771433077, 'hl1': 229, 'hl2': 234, 'dropout': 0.435084110162011}. Best is trial 11 with value: 0.6609945893287659.

# Best trial:
# R²: 0.6610
# Best params:
#   lr: 0.001763901395596742
#   hl1: 260
#   hl2: 256
#   dropout: 0.48582084114931706
