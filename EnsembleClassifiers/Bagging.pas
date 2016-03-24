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

unit Bagging;

// #############################################################
// #### Bagging bootstrap classifier
// #############################################################

interface

uses SysUtils, Classes, BaseClassifier, EnsembleClassifier;

// #############################################################
// #### Params used for the bagging learning method
type
  TVotedBaggingProps = record
    Learner : TCustomWeightedLearner;
    numRounds : integer;
    balanced : boolean;
    LearnDataSetPercentage : integer;
    BalanceOnlyOneTime : boolean;
    OwnsLearner : boolean;
  end;

// #############################################################
// #### Bagging classifier
type
  TVotedBagging = class(TEnsembelClassifier)
  public
    function Classify(Example : TCustomExample; var confidence : double) : integer; override;
  end;

// #############################################################
// #### learning a bagging classifier.
type
  TVotedBaggingLearner = class(TCustomLearner)
  private
    fProps : TVotedBaggingProps;
  protected
    function DoUnweightedLearn : TCustomClassifier; override;
  public
    procedure SetBaggingParams(const props : TVotedBaggingProps);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
    constructor Create;
    destructor Destroy; override;
  end;

implementation

uses Math;

// ##################################################################
// #### Local definitions
// ##################################################################

const cDefPercentage = 70;
      cDefBalanced = False;

{ TVotedBagging }

type
  TClassVals = record
    classVal : integer;
    numVotes : integer;
    Conf : double;
  end;

// ##################################################################
// #### Classifier
// ##################################################################

function TVotedBagging.Classify(Example : TCustomExample;
  var confidence: double): integer;
var classVals : Array of TClassVals;
    i, j : Integer;
    classVal : integer;
    conf : double;
    numClasses : integer;
    found : boolean;
    maxIdx : integer;
begin
     // #########################################################
     // #### in bagging all classifiers are evaluated. The class "wins" which
     // got the most votes (or in case of equality I define the winning class
     // as the one with the higher confidence
     SetLength(classVals, 0);
     numClasses := 0;

     for i := 0 to Classifiers.Count - 1 do
     begin
          classVal := Classifiers[i].Classify(Example, conf);
          found := False;
          for j := 0 to numClasses - 1 do
          begin
               if classVals[j].classVal = classVal then
               begin
                    found := True;
                    inc(classVals[j].numVotes);
                    classVals[j].Conf := classVals[j].Conf + conf;
                    break;
               end;
          end;

          if not found then
          begin
               if Length(classVals) <= numClasses then
               begin
                    SetLength(classVals, Max(5, numClasses*2));
                    FillChar(classVals[numClasses], sizeof(TClassVals)*(Length(classVals) - numClasses), $FF);
               end;
               
               classVals[numClasses].classVal := classVal;
               classVals[numClasses].Conf := conf;
               classVals[numClasses].numVotes := 1;
               inc(numClasses);
          end;
     end;

     // ##########################################################
     // #### evaluate votes:
     maxIdx := 0;
     for i := 1 to numClasses - 1 do
     begin
          if (classVals[i].numVotes > classVals[maxIdx].numVotes) or 
             ((classVals[i].numVotes = classVals[maxIdx].numVotes) and (classVals[i].Conf > classVals[maxIdx].Conf)) 
          then
              maxIdx := i;   
     end;

     assert(numClasses > 0, 'No classes defined');

     Result := classVals[maxIdx].classVal;
     confidence := classVals[maxIdx].Conf/classVals[maxIdx].numVotes;     
end;

{ TVotedBaggingLearner }

class function TVotedBaggingLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = TVotedBagging;
end;

constructor TVotedBaggingLearner.Create;
begin
     inherited Create;

     Randomize;
end;

destructor TVotedBaggingLearner.Destroy;
begin
     if fProps.OwnsLearner then
        fProps.Learner.Free;
        
     inherited;
end;

function TVotedBaggingLearner.DoUnweightedLearn: TCustomClassifier;
var classifiers : TClassifierSet;
    i : integer;
    trainSet : TCustomLearnerExampleList;
    classifier : TCustomClassifier;
    balancedTrainSet : TCustomLearnerExampleList;
begin
     classifiers := TClassifierSet.Create;
     try
        classifiers.OwnsObjects := True;

        balancedTrainSet := nil;
        if fProps.BalanceOnlyOneTime and fProps.balanced then
           balancedTrainSet := DataSet.CreateBalancedDataSet;

        try
           for i := 0 to fProps.NumRounds - 1 do
           begin
                // create training set
                if fProps.balanced then
                begin
                     if Assigned(balancedTrainSet)
                     then
                         trainSet := balancedTrainSet.CreateRandomDataSet(fProps.LearnDataSetPercentage)
                     else
                         trainSet := DataSet.CreateRandomizedBalancedDataSet(fProps.LearnDataSetPercentage);
                end
                else
                    TrainSet := DataSet.CreateRandomDataSet(fProps.LearnDataSetPercentage);

                try
                   assert(Assigned(TrainSet), 'error no training set created');

                   // create classifier
                   fProps.Learner.Init(TrainSet);
                   classifier := fProps.Learner.Learn;
                   classifiers.AddClassifier(classifier);
                finally
                       FreeAndNil(TrainSet);
                end;

                DoProgress(100*i div fProps.NumRounds);
           end;
        finally
               FreeAndNil(balancedTrainSet);
        end;
     except
           on E: Exception do
           begin
                FreeAndNil(classifiers);
                raise;
           end;
     end;

     // ################################################
     // #### Create the final classifier
     Result := TVotedBagging.Create(classifiers, True);
end;

procedure TVotedBaggingLearner.SetBaggingParams(const props : TVotedBaggingProps);
begin
     fProps := Props;
end;

end.
