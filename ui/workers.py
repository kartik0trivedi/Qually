import logging

from PyQt6.QtCore import QThread, pyqtSignal


gui_logger = logging.getLogger("QuallyGUI")


class ExperimentRunner(QThread):
    """Runs an experiment in a separate thread to avoid freezing the GUI."""

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, experiment_manager, experiment_id, prompt_ids):
        super().__init__()
        self.experiment_manager = experiment_manager
        self.experiment_id = experiment_id
        self.prompt_ids = prompt_ids
        self._is_running = True

    def run(self):
        """Executes the experiment run."""
        self.log_message.emit(f"Starting experiment run for ID: {self.experiment_id}")
        try:
            if not self.experiment_manager:
                raise ValueError("Experiment manager not initialized")

            experiment = self.experiment_manager.get_experiment(self.experiment_id)
            if not experiment:
                raise ValueError(f"Experiment with ID {self.experiment_id} not found")

            if not experiment.get("conditions"):
                raise ValueError("Experiment has no conditions defined")

            results = self.experiment_manager.run_experiment(self.experiment_id, self.prompt_ids)

            if results:
                self.log_message.emit(f"Experiment run completed successfully for ID: {self.experiment_id}")
                self.finished.emit(results)
            else:
                self.log_message.emit(f"Experiment run for ID {self.experiment_id} returned no results or failed.")
                self.error.emit("Experiment run finished but returned no results. Check backend logs.")

        except Exception as e:
            gui_logger.error(f"Error during experiment run thread: {e}", exc_info=True)
            self.log_message.emit(f"Error during experiment run: {e}")
            self.error.emit(f"An error occurred during the experiment: {str(e)}")

    def stop(self):
        """Stop the experiment run."""
        self._is_running = False
        self.log_message.emit("Experiment run cancellation requested.")
        # The backend does not currently support cancellation mid-run.
