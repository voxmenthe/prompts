# Project Structure

```
.
├── .cursor
│   └── rules
│       ├── agent.mdc
│       ├── environment.mdc
│       ├── files.mdc
│       ├── general_guidelines.mdc
│       ├── structure.mdc
│       └── tests.mdc
├── .gitignore
├── .scripts
│   └── update_structure.sh
├── .windsurfrules
├── README.md
├── docker-compose.yml
├── integration_tests_status.md
├── old_project_plan.md
├── option_chain_selector_plan.md
├── package_upgrade_plan.md
├── project_plan_near_term.md
├── project_plan_overall.md
├── project_plan_xai.md
├── run_app.sh
└── src
    ├── backend
    │   ├── .coverage
    │   ├── .dockerignore
    │   ├── .env.example
    │   ├── Dockerfile
    │   ├── README.md
    │   ├── app
    │   │   ├── db_upgrade.py
    │   │   ├── dependencies.py
    │   │   ├── docs
    │   │   │   └── market_data_providers.md
    │   │   ├── examples
    │   │   │   └── market_data_example.py
    │   │   ├── main.py
    │   │   ├── models
    │   │   │   ├── database.py
    │   │   │   └── schemas.py
    │   │   ├── routes
    │   │   │   ├── greeks.py
    │   │   │   ├── market_data.py
    │   │   │   ├── options.py
    │   │   │   ├── positions.py
    │   │   │   └── scenarios.py
    │   │   ├── services
    │   │   │   ├── market_data.py
    │   │   │   ├── market_data_provider.py
    │   │   │   ├── option_chain_service.py
    │   │   │   ├── option_pricing.py
    │   │   │   ├── polygon_provider.py
    │   │   │   ├── scenario_engine.py
    │   │   │   ├── volatility_service.py
    │   │   │   └── yfinance_provider.py
    │   │   └── tests
    │   │       └── test_market_data_providers.py
    │   ├── options.db
    │   ├── poetry.lock
    │   ├── project_setup.sh
    │   ├── pyproject.toml
    │   ├── pytest.ini
    │   ├── requirements.txt
    │   ├── run_all_tests.sh
    │   ├── run_test_with_mock.sh
    │   ├── run_tests.sh
    │   └── tests
    │       ├── README.md
    │       ├── __init__.py
    │       ├── conftest.py
    │       ├── integration_tests
    │       │   ├── README.md
    │       │   ├── __init__.py
    │       │   ├── conftest.py
    │       │   ├── mocks
    │       │   │   ├── __init__.py
    │       │   │   ├── polygon_api_mock.py
    │       │   │   └── redis_mock.py
    │       │   ├── mocks.py
    │       │   ├── test_database_persistence.py
    │       │   ├── test_market_data_pipeline.py
    │       │   ├── test_mock_providers.py
    │       │   └── test_strategy_pipeline.py
    │       ├── test_api_endpoints.py
    │       ├── test_database.py
    │       ├── test_health.py
    │       ├── test_market_data.py
    │       ├── test_option_api_endpoints.py
    │       ├── test_option_chain_service.py
    │       ├── test_option_pricing.py
    │       └── test_scenario_engine.py
    ├── frontend
    │   ├── app
    │   │   ├── globals.css
    │   │   ├── layout.tsx
    │   │   ├── market-data
    │   │   │   └── page.tsx
    │   │   ├── page.tsx
    │   │   ├── positions
    │   │   │   └── page.tsx
    │   │   └── visualizations
    │   │       ├── [id]
    │   │       │   └── page.tsx
    │   │       └── page.tsx
    │   ├── components
    │   │   ├── PositionForm.tsx
    │   │   └── PositionTable.tsx
    │   ├── lib
    │   │   ├── api
    │   │   │   ├── apiClient.ts
    │   │   │   ├── greeksApi.ts
    │   │   │   ├── index.ts
    │   │   │   ├── marketDataApi.ts
    │   │   │   ├── optionsApi.ts
    │   │   │   ├── positionsApi.ts
    │   │   │   └── scenariosApi.ts
    │   │   └── stores
    │   │       ├── index.ts
    │   │       ├── marketDataStore.ts
    │   │       ├── positionStore.ts
    │   │       └── scenariosStore.ts
    │   ├── next-env.d.ts
    │   ├── next.config.js
    │   ├── package-lock.json
    │   ├── package.json
    │   ├── postcss.config.js
    │   ├── tailwind.config.js
    │   └── tsconfig.json
    └── strategies
        └── bearish_pos_delta_pos_gamma_moderate_theta_debit.md

27 directories, 104 files
```
