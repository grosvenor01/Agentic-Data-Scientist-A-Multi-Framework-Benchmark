from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_version.tools.AnalysisTools import EDA
from crewai_version.tools.preprocessingTools import run_python_script_tool
from crewai_version.tools.TrainingTools import *
from crewai_version.tools.EvaluationTools import *

@CrewBase
class DataScienceCrew():
    agents: List[BaseAgent]
    tasks: List[Task]

    def supervisor(self) -> Agent:
        return Agent(
            config=self.agents_config['supervisor'],
            verbose=True,
            allow_delegation = True
        )

    @agent
    def analysis(self) -> Agent:
        return Agent(
            config=self.agents_config['analysis'],
            verbose=True,
            tools=[EDA],
            allow_delegation = True
        )
    
    @agent
    def preprocessing(self) -> Agent:
        return Agent(
            config=self.agents_config['preprocessing'],
            verbose=True,
            tools=[run_python_script_tool],
            allow_delegation = True
        )
    
    @agent
    def training(self) -> Agent:
        return Agent(
            config=self.agents_config['training'],
            verbose=True,
            tools=[
                dataLoader,
                performLinearRegression,
                performPolynomialRegression,
                performSVR,
                performGradientBoostingRegression,
                performRandomForestClassification,
                performGradientBoostingClassification,
                performLogisticRegressionClassification,
                performSVMClassification,
                performKMeansClustering,
                performPCA
            ],
            allow_delegation = True
        )
    
    @agent
    def evaluation(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluation'],
            verbose=True,
            tools=[
                perform_mae,
                perform_f1,
                perform_accuracy,
                perform_pca_variance,
                perform_r2,
                perform_rmse,
                perform_roc_auc,
                perform_silhouette,
            ],
            allow_delegation = True
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'],
        )

    @task
    def preprocessing_task(self) -> Task:
        return Task(
            config=self.tasks_config['preprocessing_task'],
            context=[self.analysis_task()]
        )
    
    @task
    def training_task(self) -> Task:
        return Task(
            config=self.tasks_config['training_task'],
            context=[self.preprocessing_task()]
        )

    @task
    def evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluation_task'],
            context=[self.training_task()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            manager_agent=self.supervisor(),
            process=Process.hierarchical,
            verbose=True,
            memory_config={
                "storage": f"./memory_jdida"
            }
        )
