# AIGC-5504-0NA-Final-Project
Synthetic Data Generation for Customer Churn Prediction
synthetic-churn-prediction/
│
├── data/                              # Data directory
│   ├── raw/                           # Original unmodified data
│   ├── processed/                     # Cleaned and preprocessed data
│   └── synthetic/                     # Generated synthetic data
│
├── src/                               # Source code
│   ├── __init__.py                    # Makes src a Python package
│   │
│   ├── data/                          # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py           # Data cleaning and preparation
│   │   ├── transformation.py          # Feature engineering and transformations
│   │   └── validation.py              # Data quality checks
│   │
│   ├── models/                        # Model implementations
│   │   ├── __init__.py
│   │   ├── vae/                       # VAE implementation
│   │   │   ├── __init__.py
│   │   │   ├── encoder.py             # VAE encoder architecture
│   │   │   ├── decoder.py             # VAE decoder architecture
│   │   │   └── vae_model.py           # Complete VAE model
│   │   │
│   │   ├── gan/                       # GAN implementation
│   │   │   ├── __init__.py
│   │   │   ├── generator.py           # GAN generator architecture
│   │   │   ├── discriminator.py       # GAN discriminator architecture
│   │   │   └── gan_model.py           # Complete GAN model
│   │   │
│   │   └── utils.py                   # Shared model utilities
│   │
│   ├── training/                      # Training procedures
│   │   ├── __init__.py
│   │   ├── vae_trainer.py             # VAE training pipeline
│   │   └── gan_trainer.py             # GAN training pipeline
│   │
│   ├── evaluation/                    # Evaluation modules
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Implementation of evaluation metrics
│   │   ├── statistical_tests.py       # Statistical similarity tests
│   │   └── ml_efficacy.py             # Machine learning efficacy tests
│   │
│   └── visualization/                 # Visualization utilities
│       ├── __init__.py
│       ├── distribution_plots.py      # Distribution comparison plots
│       └── performance_plots.py       # Model performance visualization
│
├── configs/                           # Configuration files
│   ├── data_config.json               # Data processing parameters
│   ├── vae_config.json                # VAE hyperparameters
│   └── gan_config.json                # GAN hyperparameters
│
├── scripts/                           # Executable scripts
│   ├── download_data.py               # Script to download the dataset
│   ├── preprocess_data.py             # Run data preprocessing pipeline
│   ├── train_vae.py                   # Train the VAE model
│   ├── train_gan.py                   # Train the GAN model
│   ├── generate_samples.py            # Generate synthetic samples
│   └── evaluate_models.py             # Run evaluation pipeline
│
├── requirements.txt                   # Project dependencies
├── README.md                          # Project documentation
├── .gitignore                         # Specifies files to ignore in version control
└── .env.example                       # Template for environment variables