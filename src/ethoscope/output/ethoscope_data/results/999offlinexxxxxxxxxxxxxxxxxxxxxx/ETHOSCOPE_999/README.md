The offline_tracker.py outputs rethomics-compliant results to:

    ethoscope_data/results/999offlinexxxxxxxxxxxxxxxxxxxxxx/ETHOSCOPE_999/YYYY-MM-DD_HH-MM-SS/YYYY-MM-DD_HH-MM-SS_999offlinexxxxxxxxxxxxxxxxxxxxxx.db

where YYYY-MM-DD_HH-MM-SS is the wall-clock start of the offline tracking run.
Point rethomics/scopr ``result_dir`` to ``.../output/ethoscope_data/results`` to discover these files
(e.g. ``scopr::link_ethoscope_metadata`` / ``rethomics::loadSQLite``).
