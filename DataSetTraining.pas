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

unit DataSetTraining;

// #############################################################
// #### Defines routines to split given datasets in different ways,
// #### trains a classifier and evaluates it's performance
// #############################################################

interface

uses SysUtils, Classes, BaseClassifier, Types;

type
  TTrainProgress = procedure(Sender : TObject; progress : integer; iter : integer; const TrainError, ClassError : double);

// ##########################################
// #### Learner who implements different types of splitting strategies
type
  TDataSetLearnerWeighted = class(TObject)
  private
    fDataSet : TCustomLearnerExampleList;
    fLearner : TCustomWeightedLearnerClass;
    fTrainProgress: TTrainProgress;
    fWeights : TDoubleDynArray;
  protected
    procedure DoProgress(progress : integer; iter : integer; const TrainError, ClassError : double);
  public
    property OnTrainProgress : TTrainProgress read fTrainProgress write fTrainProgress;

    // performs n-1 iterations, in each iteration one element from the dataset is dismissed and a classifier is trained
    // from the reduced list. The resulting classifier is again trained on the whole list. Returned are
    // the training and classification errors for each iteration plus the mean errors
    function LeaveOneOutLearner(const Weights : Array of double; var TrainErrors, ClassErrors : TDoubleDynArray; var MeanTrainError, MeanClassifyError : double) : TCustomClassifier; 

    constructor Create(Learner : TCustomWeightedLearnerClass; DataSet : TCustomLearnerExampleList; const Weights : TDoubleDynArray);
  end;

// ###########################################
// #### Unweighted learner are just a special case of weighted examples
type
  TDataSetLearner = class(TDataSetLearnerWeighted)
  public
    function LeaveOneOutLearner(var TrainErrors, ClassErrors : TDoubleDynArray; var MeanTrainError, MeanClassifyError : double) : TCustomClassifier;
  end;

implementation

{ TDataSetLearner }

constructor TDataSetLearnerWeighted.Create(Learner: TCustomWeightedLearnerClass;
  DataSet: TCustomLearnerExampleList; const Weights : TDoubleDynArray);
begin
     inherited Create;

     fLearner := Learner;
     fDataSet := DataSet;
     fWeights := Weights;
end;

procedure TDataSetLearnerWeighted.DoProgress(progress, iter: integer; const TrainError,
  ClassError: double);
begin
     if Assigned(fTrainProgress) then
        fTrainProgress(self, progress, iter, TrainError, ClassError);
end;

function TDataSetLearnerWeighted.LeaveOneOutLearner(
  const Weights: array of double; var TrainErrors, ClassErrors: TDoubleDynArray;
  var MeanTrainError, MeanClassifyError: double): TCustomClassifier;
var i, j, trainIdx : integer;
    leftOut : TCustomLearnerExample;
    trainList : TCustomLearnerExampleList;
    learner : TCustomWeightedLearner;
    classVal : integer;
    conf : double;
begin
     // ##############################################
     // #### The final classifier will be from the complete set
     // but the meantrain and classify errors are built from the
     // leave one out algorithm
     MeanTrainError := 0;
     MeanClassifyError := 0;
     SetLength(TrainErrors, fDataSet.Count);
     SetLength(ClassErrors, fDataSet.Count);
     leftOut := nil;

     // ##############################################
     // #### For each classifier leave one exmaple out
     trainList := TCustomLearnerExampleList.Create;
     try
        for i := 0 to fDataSet.Count - 1 do
        begin
             trainList.Clear;
             TrainErrors[i] := 0;
             ClassErrors[i] := 0;
             for j := 0 to fDataSet.Count - 1 do
             begin
                  if i = j
                  then
                      leftOut := fDataSet[j]
                  else
                      trainList.Add(fDataSet[j]);
             end;

             // ############################################
             // #### Create classifier
             learner := fLearner.Create;
             try
                learner.Init(trainList);
                Result := learner.Learn(Weights);
                try
                   // ############################################
                   // #### Check out errors - weighted!
                   for trainIdx := 0 to trainList.Count - 1 do
                   begin
                        classVal := Result.Classify(trainList[trainIdx], conf);
                        if classVal <> trainList[trainIdx].ClassVal then
                           TrainErrors[i] := TrainErrors[i] + Weights[i];
                   end;

                   classVal := Result.Classify(leftOut, conf);
                   if classVal <> leftOut.ClassVal then
                      ClassErrors[i] := Weights[i];
                finally
                       Result.Free;
                end;
             finally
                    learner.Free;
             end;

             // #######################################
             // #### update mean values
             MeanTrainError := MeanTrainError + TrainErrors[i];
             MeanClassifyError := MeanClassifyError +  ClassErrors[i];

             // inform caller.
             DoProgress((i + 1)*100 div fDataSet.Count, i + 1, TrainErrors[i], ClassErrors[i]);
        end;
     finally
            trainList.Free;
     end;

     // ############################################
     // #### Finally create a classifier from the whole list
     learner := fLearner.Create;
     try
        learner.Init(fDataSet);
        Result := learner.Learn(Weights);
     finally
            learner.Free;
     end;
end;

function TDataSetLearner.LeaveOneOutLearner(var TrainErrors,
  ClassErrors: TDoubleDynArray; var MeanTrainError,
  MeanClassifyError: double): TCustomClassifier;
var weights : Array of double;
    i : integer;
begin
     SetLength(weights, fDataSet.Count - 1);
     for i := 0 to Length(weights) - 1 do
         weights[i] := 1/Length(weights);

     Result := inherited LeaveOneOutLearner(Weights, TrainErrors, ClassErrors, MeanTrainError, MeanClassifyError);
end;

end.
