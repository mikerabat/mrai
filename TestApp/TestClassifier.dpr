// ###################################################################
// #### This file is part of the artificial intelligence project, and is
// #### offered under the licence agreement described on
// #### http://www.mrsoft.org/
// ####
// #### Copyright:(c) 2014, Michael R. . All rights reserved.
// ####
// #### Unless required by applicable law or agreed to in writing, software
// #### distributed under the License is distributed on an "AS IS" BASIS,
// #### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// #### See the License for the specific language governing permissions and
// #### limitations under the License.
// ###################################################################

program TestClassifier;

uses
  FastMM4 in 'D:\Daten\Delphi2010\3rdParty\FastMM4\FastMM4.pas',
  Forms,
  ufrmTestClassifier in 'ufrmTestClassifier.pas' {frmClassifierTest},
  ImageDataSet in '..\DataIO\ImageDataSet.pas',
  FischerBatchLDA in '..\SimpleClassifier\FischerBatchLDA.pas',
  FischerClassifiers in '..\SimpleClassifier\FischerClassifiers.pas',
  IncrementalFischerLDA in '..\IncrementalClassifier\IncrementalFischerLDA.pas',
  FischerIncrementalClassifiers in '..\IncrementalClassifier\FischerIncrementalClassifiers.pas',
  AdaBoost in '..\EnsembleClassifiers\AdaBoost.pas',
  Bagging in '..\EnsembleClassifiers\Bagging.pas',
  CustomBooster in '..\EnsembleClassifiers\CustomBooster.pas',
  IncrementalImageDataSet in '..\DataIO\IncrementalImageDataSet.pas',
  IntegralImg in '..\FeatureExtractors\IntegralImg.pas',
  Haar2D in '..\FeatureExtractors\Haar2D.pas',
  Haar1D in '..\FeatureExtractors\Haar1D.pas',
  Haar1DLearner in '..\FeatureExtractors\Haar1DLearner.pas',
  Haar2DAdaBoost in '..\EnsembleClassifiers\Haar2DAdaBoost.pas',
  BoostCascade in '..\EnsembleClassifiers\BoostCascade.pas',
  Image2DSweep in '..\Image2DSweep.pas',
  Haar2DImageSweep in '..\Haar2DImageSweep.pas',
  Haar2DDataSet in '..\DataIO\Haar2DDataSet.pas',
  SimpleDecisionStump in '..\SimpleClassifier\SimpleDecisionStump.pas',
  DecisionTree45 in '..\SimpleClassifier\DecisionTree45.pas',
  TreeStructs in '..\SimpleClassifier\TreeStructs.pas',
  NaiveBayes in '..\SimpleClassifier\NaiveBayes.pas',
  SVM in '..\SimpleClassifier\SVM.pas',
  BaseClassifier in '..\BaseClassifier.pas',
  BaseIncrementalLearner in '..\BaseIncrementalLearner.pas',
  BaseMatrixExamples in '..\BaseMatrixExamples.pas',
  ClassifierUtils in '..\ClassifierUtils.pas',
  DataSetTraining in '..\DataSetTraining.pas',
  EnsembleClassifier in '..\EnsembleClassifier.pas',
  RBF in '..\SimpleClassifier\RBF.pas',
  FastMM4Messages in 'D:\Daten\Delphi2010\3rdParty\FastMM4\FastMM4Messages.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TfrmClassifierTest, frmClassifierTest);
  Application.Run;
end.
