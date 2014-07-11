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

unit AdaBoost;

// ####################################################
// #### Classifier and learner of the base adaboost algorithm
// ####################################################

// paper: Additive Logistic Regression: a Statistical View of Boosting
// Friedman, Hastie, Tibshirani

interface

uses SysUtils, Classes, BaseClassifier, EnsembleClassifier, CustomBooster, Types;

type
  TCustomBoostPruner = class(TCustomBoostLearner)
  protected
    procedure PruneToLowestTrainError(Classifiers : TClassifierSet; const weights : Array of double); override;
  end;

// AdaBoost classification function -> Results in -1, 1
type
  TDiscreteAdaBoostClassifier = class(TCustomBoostingClassifier)
  public
    function Classify(Example : TCustomExample; var confidence : double) : integer; override;
  end;

// ####################################################
// #### Base adaboost algorithm
type
  TDiscreteAdaBoostLearner = class(TCustomBoostPruner)
  protected
    function BoostRound(var weakClassifier : TCustomClassifier; var Weights: TDoubleDynArray; var classifierWeight: double): boolean; override;
  public
    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
    class function BoostClass : TCustomBoostingClassifierClass; override;
  end;

// ####################################################
// #### Gentle boosting
// note: for gentle boosting one must have a classifier who ensures that the confidence realy returns
// a probabilty. 
type
  TGentleBoostClassifier = class(TCustomBoostingClassifier)
  public
    function Classify(Example : TCustomExample; var confidence : double) : integer; override;
  end;

// ####################################################
// #### Base gentle boost algorithm
type
  TGentleBoostLearner = class(TCustomBoostPruner)
  protected
    function BoostRound(var classifier : TCustomClassifier; var Weights: TDoubleDynArray; var classifierWeight: double): boolean; override;
  public
    class function BoostClass : TCustomBoostingClassifierClass; override;
    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
  end;

implementation

uses Math, BaseMathPersistence;

{ TBinaryClassAdaBoostLearner }

class function TDiscreteAdaBoostLearner.BoostClass: TCustomBoostingClassifierClass;
begin
     Result := TDiscreteAdaBoostClassifier;
end;

function TDiscreteAdaBoostLearner.BoostRound(var weakClassifier : TCustomClassifier; var Weights: TDoubleDynArray; var classifierWeight: double): boolean;
var numCorrectClassifications : integer;
    i : integer;
    classVals : Array of integer;
    confidence : double;
    beta : double;
begin
     Result := True;

     SetLength(classVals, Length(weights));

     // create a classifier from the given data set and the current weights
     weakClassifier := Props.WeakLearner.Learn(Weights);

     // evaluate the classifiers performance
     numCorrectClassifications := 0;
     for i := 0 to DataSet.Count - 1 do
     begin
          classVals[i] := weakClassifier.Classify(DataSet[i], confidence);

          if ClassVals[i] = DataSet[i].ClassVal then
             inc(numCorrectClassifications);
     end;

     assert(numCorrectClassifications <> 0, 'Error no classifier may classify everything wrong!');

     if DataSet.Count = numCorrectClassifications then
     begin
          // we found a perfect classifier -> use only this one:
          classifierWeight := 1;

          Result := False;
     end
     else
     begin
          beta := (DataSet.Count - numCorrectClassifications)/DataSet.Count;
          if beta > 0.5 then
          begin
               // the classifier cannot do better
               // todo: eventualy reverse classifier polarity which but that only works for 2 class problems
               FreeAndNil(weakClassifier);
               Result := False;
               exit;
          end;
          beta := beta/(1 - beta);

          classifierWeight := ln(1/beta);

          // reweight the current example weights:
          for i := 0 to DataSet.Count - 1 do
          begin
               if classVals[i] = DataSet[i].ClassVal then
                  Weights[i] := Weights[i]*beta;
          end;
     end;
end;

class function TDiscreteAdaBoostLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = TDiscreteAdaBoostClassifier;
end;

{ TBinaryClassAdaBoostClassifier }

function TDiscreteAdaBoostClassifier.Classify(Example: TCustomExample;
  var confidence: double): integer;
var classifySum : double;
    k : integer;
begin
     classifySum := fB;

     for k := 0 to Classifiers.Count - 1 do
         classifySum := classifySum + fWeights[k]*Classifiers[k].Classify(Example);

     if classifySum >= 0
     then
         Result := 1
     else
         Result := -1;

     // todo: should be definitly something between 0, 1
     confidence := Abs(classifySum);
end;

{ TGentleBoostLearner }

class function TGentleBoostLearner.BoostClass: TCustomBoostingClassifierClass;
begin
     Result := TGentleBoostClassifier;
end;

function TGentleBoostLearner.BoostRound(var classifier: TCustomClassifier;
  var Weights: TDoubleDynArray; var classifierWeight: double): boolean;
var i : integer;
    classVals : Array of integer;
    confidence : Array of double;
begin
     Result := True;

     SetLength(classVals, Length(weights));
     SetLength(confidence, Length(weights));

     // create a classifier from the given data set and the current weights
     classifier := Props.WeakLearner.Learn(Weights);

     // evaluate the classifiers performance
     for i := 0 to DataSet.Count - 1 do
         classVals[i] := classifier.Classify(DataSet[i], confidence[i]);

     classifierWeight := 1;  // note the classifier uses the confidence anyway...

     // reweight the current example weights:
     for i := 0 to DataSet.Count - 1 do
         Weights[i] := Weights[i]*exp(-DataSet[i].ClassVal*classvals[i]*confidence[i]);
end;

class function TGentleBoostLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = TGentleBoostClassifier;
end;

{ TGentleBoostClassifier }

function TGentleBoostClassifier.Classify(Example: TCustomExample;
  var confidence: double): integer;
var i : integer;
    actConf : double;
    classVal : integer;
begin
     confidence := fB;
     for i := 0 to Classifiers.Count - 1 do
     begin
          classVal := Classifiers[i].Classify(Example, actConf);
          confidence := confidence + classVal*actConf;
     end;

     Result := ifthen(confidence > 0, 1, -1);

     confidence := Abs(confidence);
end;

{ TCustomBoostPruner }

procedure TCustomBoostPruner.PruneToLowestTrainError(
  Classifiers: TClassifierSet; const weights: array of double);
var i : integer;
    j : integer;
    k : integer;
    numCorrect : integer;
    bestCorrect : integer;
    lowestIdx : integer;
    classifier : TCustomBoostingClassifier;
    helperSet : TClassifierSet;
    conf : double;
    obj : TObject;
begin
     lowestIdx := 0;
     bestCorrect := 0;
     helperSet := TClassifierSet.Create;
     try
        helperSet.OwnsObjects := False;

        for i := 0 to Classifiers.Count - 1 do
        begin
             helperSet.Clear;
             for k := 0 to i do
                 helperSet.Add(Classifiers[k]);

             classifier := BoostClass.Create(helperSet, False, weights);
             try
                numCorrect := 0;
                for j := 0 to DataSet.Count - 1 do
                begin
                     // #############################################
                     // #### classify all examples using a subset of classifiers
                     if classifier.Classify(DataSet[j], conf) = DataSet[j].ClassVal then
                        inc(numCorrect);
                end;

                if numCorrect > bestCorrect then
                begin
                     lowestIdx := i;
                     bestCorrect := numCorrect;
                end;
             finally
                    classifier.Free;
             end;
        end;
     finally
            helperSet.Free;
     end;

     // ########################################################
     // #### delete all classifiers which made the result worse
     for i := Classifiers.Count - 1 downto lowestIdx + 1 do
     begin
          obj := Classifiers[i];
          classifiers.Delete(i);
          if not classifiers.OwnsObjects then
             obj.Free;
     end;
end;

initialization
  RegisterMathIO(TDiscreteAdaBoostClassifier);
  RegisterMathIO(TGentleBoostClassifier);

end.
