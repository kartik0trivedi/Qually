import logging

from PyQt6.QtCore import QThread, pyqtSignal


gui_logger = logging.getLogger("QuallyGUI")


class ExperimentRunner(QThread):
    """Runs an experiment in a separate thread to avoid freezing the GUI."""

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, experiment_manager, experiment_ids, prompt_ids, num_runs=1):
        super().__init__()
        self.experiment_manager = experiment_manager
        self.experiment_ids = list(experiment_ids)
        self.prompt_ids = prompt_ids
        self.num_runs = max(1, int(num_runs))
        self._is_running = True

    def run(self):
        """Executes the experiment run."""
        self.log_message.emit(f"Starting experiment run for IDs: {', '.join(self.experiment_ids)}")
        try:
            if not self.experiment_manager:
                raise ValueError("Experiment manager not initialized")

            total_calls = 0
            for experiment_id in self.experiment_ids:
                experiment = self.experiment_manager.get_experiment(experiment_id)
                if not experiment:
                    raise ValueError(f"Experiment with ID {experiment_id} not found")

                if not experiment.get("conditions"):
                    raise ValueError(f"Experiment {experiment_id} has no conditions defined")
                total_calls += len(experiment.get("conditions", [])) * len(self.prompt_ids) * self.num_runs

            all_results = []
            completed_calls = 0

            def emit_progress(current, _total):
                self.progress.emit(completed_calls + current, total_calls)

            for experiment_id in self.experiment_ids:
                if not self._is_running:
                    break

                results = self.experiment_manager.run_experiment(
                    experiment_id,
                    self.prompt_ids,
                    self.num_runs,
                    progress_callback=emit_progress,
                )

                if results:
                    all_results.append(results)
                    experiment = self.experiment_manager.get_experiment(experiment_id) or {}
                    completed_calls += len(experiment.get("conditions", [])) * len(self.prompt_ids) * self.num_runs
                    self.progress.emit(completed_calls, total_calls)
                    self.log_message.emit(f"Experiment run completed successfully for ID: {experiment_id}")
                else:
                    self.log_message.emit(f"Experiment run for ID {experiment_id} returned no results or failed.")
                    self.error.emit("Experiment run finished but returned no results. Check backend logs.")
                    return

            self.finished.emit(all_results)

        except Exception as e:
            gui_logger.error(f"Error during experiment run thread: {e}", exc_info=True)
            self.log_message.emit(f"Error during experiment run: {e}")
            self.error.emit(f"An error occurred during the experiment: {str(e)}")

    def stop(self):
        """Stop the experiment run."""
        self._is_running = False
        self.log_message.emit("Experiment run cancellation requested.")
        # The backend does not currently support cancellation mid-run.
