from utils import *

class SETFIT(object):
    def __init__(self, 
    n_classes,
    batch_size,
    num_epochs, 
    learning_rate,
    SEED=SEED
    ):
        self.N_CLASSES = n_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.SEED = SEED

        path_model = 'bert-base-spanish-wwm-cased-xnli'
        #path_model = 'distilbert-base-es-multilingual-cased'
        
        self.model = SetFitModel.from_pretrained(
            os.path.join(os.getcwd(), 'out', path_model),
            )

    def fit(self, X, y, X_val, y_val):
        df = pd.DataFrame()
        df['text'] = copy.deepcopy(X)
        df['label'] = copy.deepcopy( y )
        train_dataset = Dataset.from_pandas( df )
        del df
        gc.collect()
        df = pd.DataFrame()
        df['text'] = copy.deepcopy(X_val)
        df['label'] = copy.deepcopy( y_val )
        eval_dataset = Dataset.from_pandas( df )
        del df
        gc.collect()

        args = TrainingArguments(
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            loss=CosineSimilarityLoss,
            num_iterations=20,
            head_learning_rate = self.learning_rate,
            seed = self.SEED,
            show_progress_bar=False,
        )
        args.eval_strategy = args.evaluation_strategy

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metric="accuracy",
        )
        self.trainer.remove_callback(PrinterCallback)

        self.trainer.train()

    def predict(self, X):
        df = pd.DataFrame()
        df['text'] = copy.deepcopy(X)
        test_dataset = Dataset.from_pandas( df )
        del df
        gc.collect()
        return self.trainer.model.predict(test_dataset['text']).numpy()

    def predict_proba(self, X):
        df = pd.DataFrame()
        df['text'] = copy.deepcopy(X)
        test_dataset = Dataset.from_pandas( df )
        del df
        gc.collect()
        return self.trainer.model.predict_proba(test_dataset['text']).numpy()

