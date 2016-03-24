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

unit DecisionTree45;

// #############################################################
// #### Implementation of a simple
// #### the C4.5 decision tree with weighted examples
// #############################################################

interface

uses SysUtils, Classes, BaseClassifier, TreeStructs, Types;

type
  T45TreeLearnType = (ltFull, ltMaxDepth, ltPrune);

type
  TC45Props = record
    case LearnType : T45TreeLearnType of
      ltMaxDepth: (MaxDepth : integer);
      ltPrune: ( UseValidationSet : boolean; ValidationsetSize : double);
  end;

type
  T45NodeData = class(TObject)
  public
    NumClasses : integer;
    Classes : TIntegerDynArray;
    NumExamples : integer;
    WeightedSums : TDoubleDynArray;
    FeatureSplitIndex : integer;
    SplitVal : double;

    procedure WriteToStream(Stream : TStream);
    procedure LoadFromStream(stream : TStream);

    procedure Clear;

    constructor Create;
  end;

type
  TC45Classifier = class(TCustomClassifier)
  private
    fTree : TCustomTreeItem;
    fClasses : TIntegerDynArray;

    procedure SaveTreeToStream(stream : TStream);
    procedure LoadTreeFromStream(stream : TStream);
  protected
    property Tree : TCustomTreeItem read fTree;
  public
    function Classify(Example : TCustomExample; var confidence : double) : integer; override;

    procedure OnLoadBinaryProperty(const Name : String; const Value; size : integer); override;
    procedure DefineProps; override;

    constructor Create(Tree : TCustomTreeItem);
    destructor Destroy; override;
  end;

// ###########################################################
// #### Learning C4.5 decission tree:
// http://www.cis.temple.edu/~ingargio/cis587/readings/id3-c45.html
type
  TC45Learner = class(TCustomWeightedLearner)
  private
    fProps : TC45Props;
    fLastFeatureIdx : integer;

    procedure PruneTree(tree : TCustomTreeItem; var dataSetIdx : TIntegerDynArray);
    function SplitByProperty(const Weights : Array of double; var dataSetIdx : TIntegerDynArray; curDepth : integer) : TCustomTreeItem;
    function CalculateGain(const Weights : Array of double; const dataSetIdx : TIntegerDynArray; FeatureIdx : integer;
                           var SplitVal : double; setEntropy : double; numClasses : integer; var numElements : integer) : double;
  protected
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    procedure SetProperties(const Props : TC45Props);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
  end;

implementation

uses Math, MathUtilFunc, BaseMathPersistence;

const eps = 1e-6;
      c45ClassesProps = 'classes';
      c45TreeProps = 'treedata';

procedure T45NodeData.Clear;
begin
     NumClasses := 0;
     Classes := nil;
     WeightedSums := nil;
end;

constructor T45NodeData.Create;
begin
     NumClasses := 0;
     Classes := nil;
     WeightedSums := nil;
     FeatureSplitIndex := -1;
     SplitVal := 0;
end;

procedure T45NodeData.LoadFromStream(stream: TStream);
var len : integer;
begin
     Stream.ReadBuffer(NumClasses, sizeof(integer));
     Stream.ReadBuffer(len, sizeof(integer));
     if len > 0 then
     begin
          SetLength(classes, len);
          Stream.ReadBuffer(classes[0], len*sizeof(integer));
     end;
     Stream.ReadBuffer(NumExamples, sizeof(integer));
     Stream.ReadBuffer(len, sizeof(integer));
     if len > 0 then
     begin
          SetLength(WeightedSums, len);
          Stream.WriteBuffer(WeightedSums[0], len*sizeof(double));
     end;
     Stream.ReadBuffer(FeatureSplitIndex, sizeof(integer));
     Stream.ReadBuffer(SplitVal, sizeof(double));
end;

procedure T45NodeData.WriteToStream(Stream: TStream);
var len : integer;
begin
     Stream.WriteBuffer(NumClasses, sizeof(integer));
     len := Length(Classes);
     Stream.WriteBuffer(len, sizeof(integer));
     if len > 0 then
        Stream.WriteBuffer(classes[0], len*sizeof(integer));
     Stream.WriteBuffer(NumExamples, sizeof(integer));
     len := Length(WeightedSums);
     Stream.WriteBuffer(len, sizeof(integer));
     if len > 0 then
        Stream.WriteBuffer(WeightedSums[0], len*sizeof(double));
     Stream.WriteBuffer(FeatureSplitIndex, sizeof(integer));
     Stream.WriteBuffer(SplitVal, sizeof(double));
end;

{ T45DecisionTreeLearner }

function TC45Learner.CalculateGain(const Weights: array of double;
  const dataSetIdx: TIntegerDynArray; FeatureIdx: integer;
  var SplitVal: double; setEntropy : double; numClasses : integer; var numElements : integer): double;
var i : integer;
    classes : Array of integer;
    k : Integer;
    sortIdx : TIntegerDynArray;
    sumWeight : double;
    sumLeft : Array of double;
    sumRight : Array of double;
    maxEntropy : double;
    entropy : double;
    sumWeightLeft : Double;
    entropyGain : double;
begin
     // calculatinoas the information gain IG(Ex, a) = H(Ex) - H(Ex|a)
     // where H(Ex) is the entropy of the complete set and H(Ex|a) the entropy
     // of the examples

     SplitVal := 0;

     sortIdx := CalcSortIdx(dataSetIdx, FeatureIdx);

     sumWeight := 0;
     for i := 0 to Length(dataSetIdx) - 1 do
         sumWeight := sumWeight + weights[dataSetIdx[i]];

     // find entry which minimizes the entropy gain if splitted by current attribute
     SetLength(sumLeft, numClasses);
     SetLength(sumRight, numClasses);
     SetLength(classes, numClasses);

     for i := 0 to numClasses - 1 do
     begin
          sumLeft[i] := 0;
          sumRight[i] := 0;
          classes[i] := 0;
     end;

     // initialize the "right" side
     for i := 0 to Length(dataSetIdx) - 1 do
     begin
          for k := 0 to numClasses - 1 do
          begin
               if DataSet[DataSetIdx[i]].ClassVal = classes[k]
               then
                   sumRight[k] := sumRight[k] + Weights[dataSetIdx[i]]
               else if classes[k] = 0 then
               begin
                    sumRight[k] := Weights[dataSetIdx[i]];
                    classes[k] := DataSet[DataSetIdx[i]].ClassVal;
               end;
          end;
     end;

     maxEntropy := -MaxDouble;
     SplitVal := 0;
     numElements := 0;
     sumWeightLeft := 0;

     // now go through all splits and update the "left" side -> find the minimum entropy
     for i := 0 to Length(dataSetIdx) - 2 do
     begin
          // move current element to the "left"
          for k := 0 to numClasses - 1 do
          begin
               if DataSet[DataSetIdx[sortIdx[i]]].ClassVal = classes[k] then
               begin
                    sumLeft[k] := sumLeft[k] + weights[dataSetIdx[sortIdx[i]]];
                    sumRight[k] := sumRight[k] - weights[dataSetIdx[sortIdx[i]]];
                    sumWeightLeft := sumWeightLeft + weights[dataSetIdx[sortIdx[i]]];
                    break;
               end;
          end;

          // calc overall entropy for both datasets
          entropy := 0;
          for k := 0 to numClasses - 1 do
          begin
               if sumLeft[k] > 0 then
                  entropy := entropy - sumLeft[k]/sumWeightLeft*log2(sumLeft[k]/sumWeightLeft);
               if sumRight[k] > 0 then
                  entropy := entropy - sumRight[k]/(sumWeight - sumWeightLeft)*log2(sumRight[k]/(sumWeight - sumWeightLeft));
          end;

          // todo: better a real gain or the difference
          //entropyGain := entropy/setEntropy;
          entropyGain := setEntropy - entropy;


          if entropyGain > maxEntropy then
          begin
               numElements := i + 1;
               maxEntropy := entropyGain;
               SplitVal := (DataSet[DataSetIdx[sortIdx[i]]].FeatureVec[FeatureIdx] +
                            DataSet[DataSetIdx[sortIdx[i + 1]]].FeatureVec[FeatureIdx]) / 2;
          end;
     end;

     Result := maxEntropy;
end;

class function TC45Learner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = TC45Classifier;
end;

// learning a C4.5 classifier: http://www.cis.temple.edu/~ingargio/cis587/readings/id3-c45.html

function TC45Learner.DoLearn(
  const weights: array of double): TCustomClassifier;
var tree : TCustomTreeItem;
    i : integer;
    trainingIdx, validationIdx : TIntegerDynArray;
    trainLen : integer;
    hlp : integer;
    idx : integer;
begin
     SetLength(trainingIdx, Length(weights));

     for i := 0 to Length(weights) - 1 do
         trainingIdx[i] := i;

     if (fProps.LearnType = ltPrune) and (fProps.UseValidationSet) and (fProps.ValidationsetSize > 0) then
     begin
          // Create shuffle set
          for i := length(weights) - 1 downto 1 do
          begin
               idx := random(i + 1);

               hlp := trainingIdx[idx];
               trainingIdx[idx] := trainingIdx[i];
               trainingIdx[i] := hlp;
          end;
          trainLen := Max(1, Round(Length(weights)*(1 - fProps.ValidationsetSize)));
          SetLength(validationIdx, Length(weights) - trainLen);
          for i := trainLen to Length(weights) - 1 do
              validationIdx[i - trainLen] := trainingIdx[i];
          SetLength(trainingIdx, trainLen);
     end
     else
         validationIdx := trainingIdx;

     tree := SplitByProperty(weights, trainingIdx, 0);

     if fProps.LearnType = ltPrune then
        PruneTree(tree, validationIdx);

     Result := TC45Classifier.Create(tree);
end;

// pruning see: http://ai.stanford.edu/~ronnyk/treesHB.pdf
// base idea used here: reduced error pruning
// -> if a node can be pruned without raising the error replace it
procedure TC45Learner.PruneTree(tree: TCustomTreeItem; var dataSetIdx : TIntegerDynArray);
var rootNode : TTreeNode;
    rootCoorect : integer;
    numClassItems : integer;
    i : Integer;
    nodeData : T45NodeData;
    splitData : TIntegerDynArray;
    leave : TTreeLeave;
procedure SplitDataSet(var dataSetIdx, splitIdx : TIntegerDynArray);
var numItems : integer;
    i : integer;
    data : T45NodeData;
    numDataIdx : integer;
begin
     data := T45NodeData(rootNode.TreeData);
     SetLength(splitIdx, Length(dataSetIdx));
     numItems := 0;
     numDataIdx := 0;

     for i := 0 to Length(dataSetIdx) - 1 do
     begin
          if (DataSet[dataSetIdx[i]].FeatureVec[data.FeatureSplitIndex] < data.SplitVal) then
          begin
               splitIdx[numItems] := dataSetIdx[i];
               inc(numItems);
          end
          else
          begin
               dataSetIdx[numDataIdx] := dataSetIdx[i];
               inc(numDataIdx);
          end;
     end;

     SetLength(splitIdx, numItems);
     SetLength(dataSetIdx, numDataIdx);
end;

function TreeClassify(node : TCustomTreeItem; idx : integer) : integer;
var nodeData : T45NodeData;
begin
     while Assigned(node) and not (node is TTreeLeave) do
     begin
          nodeData := T45NodeData(node.TreeData);

          if DataSet[idx].FeatureVec[nodeData.FeatureSplitIndex] < nodeData.SplitVal
          then
              node := TTreeNode(node).LeftItem
          else
              node := TTreeNode(node).RightItem;

          assert(Assigned(node), 'No leave assigned');
     end;

     nodeData := T45NodeData(node.TreeData);
     Result := nodeData.Classes[0];
end;

function calcError(tree : TTreeNode; dataSetIdx : TIntegerDynArray) : integer;
var i : integer;
    numCorrect : integer;
begin
     numCorrect := 0;
     for i := 0 to Length(dataSetIdx) - 1 do
         if TreeClassify(Tree, dataSetIdx[i]) = DataSet[dataSetIdx[i]].ClassVal then
            inc(numCorrect);

     Result := numCorrect;
end;
var cl1, cl2 : integer;
begin
     // recursive implementation of the pruning algorithm
     if (tree is TTreeLeave) then
        exit;

     // ##################################################
     // #### Prune tree according to the current node
     rootNode := tree as TTreeNode;
     rootCoorect := calcError(rootNode, dataSetIdx);

     numClassItems := 0;
     cl1 := DataSet[dataSetIdx[0]].ClassVal;
     cl2 := -cl1;
     for i := 0 to Length(dataSetIdx) - 1 do
         if DataSet[dataSetIdx[i]].ClassVal = DataSet[dataSetIdx[0]].ClassVal
         then
             inc(numClassItems)
         else
             cl2 := DataSet[dataSetIdx[i]].ClassVal;

     SplitDataSet(dataSetIdx, splitData);

     if rootCoorect >= Max(Length(dataSetIdx) - numClassItems, numClassItems) then
     begin
          if Length(splitData) > 0 then
             PruneTree(rootNode.LeftItem, splitData);
          if Length(dataSetIdx) > 0 then
             PruneTree(rootNode.RightItem, dataSetIdx);
     end
     else
     begin
          // merge the datasets again
          SetLength(dataSetIdx, Length(dataSetIdx) + Length(splitData));
          if Length(splitData) > 0 then
             Move(splitData[0], dataSetIdx[Length(dataSetIdx) - Length(splitData)], Length(splitData)*sizeof(splitData[0]));

          // exchange the current node by the leave with the class with the most occurances
          // substitute root node by a leave
          if Assigned(rootNode.Parent) then
          begin
               leave := TTreeLeave.Create;
               nodeData := T45NodeData.Create;
               nodeData.SplitVal := 0;
               nodeData.FeatureSplitIndex := -1;
               SetLength(nodeData.Classes, 1);
               nodeData.Classes[0] := ifthen(Length(dataSetIdx) - numClassItems < numClassItems, cl1, cl2);

               leave.TreeData := nodeData;

               if TTreeNode(rootNode.Parent).LeftItem = rootNode then
               begin
                    TTreeNode(rootNode.Parent).LeftItem.Free;
                    TTreeNode(rootNode.Parent).LeftItem := leave;
               end
               else
               begin
                    TTreeNode(rootNode.Parent).RightItem.Free;
                    TTreeNode(rootNode.Parent).RightItem := leave;
               end;
          end;
     end;
end;

procedure TC45Learner.SetProperties(
  const Props: TC45Props);
begin
     fProps := Props;
end;

function TC45Learner.SplitByProperty(const Weights : Array of double; var dataSetIdx : TIntegerDynArray; curDepth : integer): TCustomTreeItem;
var i, j : integer;
    Classes : TIntegerDynArray;
    WeightedNumElements : TDoubleDynArray;
    NumExamples : TIntegerDynArray;
    overallWeight : double;
    numClasses : integer;
    clFound : boolean;
    clVal : integer;
    entropy : double;
    prob : double;
    maxGain : double;
    splitVal : double;
    maxSplitVal : double;
    gain : double;
    maxSplitFeatureIdx : integer;
    maxnumElements : integer;
    numElements : integer;
    nodeData : T45NodeData;
    leftData : TIntegerDynArray;
    rightData : TIntegerDynArray;
    leftIdx, rightIdx : integer;
    tempD : double;
    tempI : integer;
    overallNumExamples : integer;
begin
     assert(Length(dataSetIdx) > 0, 'Error empty dataset for splitting');

     // find the property with the highest information gain -> split there.
     SetLength(WeightedNumElements, 10);
     SetLength(NumExamples, 10);
     SetLength(Classes, 10);
     numClasses := 0;
     overallWeight := 0;
     overallNumExamples := 0;

     // calculate statistics
     for i := 0 to Length(dataSetIdx) - 1 do
     begin
          clVal := DataSet[dataSetIdx[i]].ClassVal;
          clFound := False;

          for j := 0 to numClasses - 1 do
          begin
               if Classes[j] = clVal then
               begin
                    clFound := True;
                    WeightedNumElements[j] := weightedNumElements[j] + weights[dataSetIdx[i]];
                    inc(NumExamples[j]);
                    break;
               end;
          end;

          if not clFound then
          begin
               if Length(Classes) <= numClasses + 1 then
               begin
                    SetLength(Classes, Min(Length(classes) + 1024, 2*numClasses));
                    SetLength(weightedNumElements, Length(Classes));
                    SetLength(NumExamples, Length(Classes));
               end;

               Classes[NumClasses] := clVal;
               weightedNumElements[NumClasses] := weights[dataSetIdx[i]];
               NumExamples[NumClasses] := 1;
               inc(numClasses);
          end;

          overallWeight := overallWeight + weights[dataSetIdx[i]];
     end;

     for i := 0 to numClasses - 1 do
         inc(overallNumExamples, NumExamples[i]);

     // check the max depth property:
     if (fProps.LearnType = ltMaxDepth) and (curDepth = fProps.MaxDepth) then
     begin
          // calculate overall entropy or "impurity" of that node
          entropy := 0;
          for i := 0 to numClasses - 1 do
          begin
               prob := weightedNumElements[i]/(overallWeight + eps);
               entropy := entropy - prob*Log2(prob);
          end;

          // assign a leave node to the class with the best performance
          nodeData := T45NodeData.Create;
          nodeData.NumClasses := 1;
          nodeData.Classes := Copy(Classes, 0, NumClasses);
          nodeData.WeightedSums := Copy(WeightedNumElements, 0, NumClasses);
          nodeData.FeatureSplitIndex := -1;
          nodeData.SplitVal := 0;
          nodeData.NumExamples := overallNumExamples;

          for i := 1 to numClasses - 1 do
          begin
               if nodeData.WeightedSums[i] > nodeData.WeightedSums[0] then
               begin
                    tempi := nodeData.Classes[i];
                    nodeData.Classes[i] := nodeData.Classes[0];
                    nodeData.Classes[0] := tempi;

                    tempD := nodeData.WeightedSums[i];
                    nodeData.WeightedSums[i] := nodeData.WeightedSums[0];
                    nodeData.WeightedSums[0] := tempD;
               end;
          end;

          Result := TTreeLeave.Create;
          Result.TreeData := nodeData;
          exit;
     end;

     // ###########################################################
     // #### split with the normal procedure
     if numClasses = 1 then
     begin
          nodeData := T45NodeData.Create;
          nodeData.NumClasses := 1;
          nodeData.Classes := Copy(Classes, 0, numClasses);
          nodeData.FeatureSplitIndex := -1;
          nodeData.SplitVal := 0;
          nodeData.NumExamples := overallNumExamples;

          // we are finished and ended up in a leave...
          Result := TTreeLeave.Create;
          Result.TreeData := nodeData;
     end
     else
     begin
          // ###########################################################
          // #### Split the dataset:

          // calculate overall entropy or "impurity" of that node
          entropy := 0;
          for i := 0 to numClasses - 1 do
          begin
               prob := weightedNumElements[i]/(overallWeight + eps);
               entropy := entropy - prob*Log2(prob);
          end;

          // calculate information gain if one would split the node at this property
          maxGain := -MaxDouble;
          maxSplitVal := 0;
          maxSplitFeatureIdx := 0;
          numElements := 0;
          maxnumElements := 0;

          for i := 0 to DataSet[0].FeatureVec.FeatureVecLen - 1 do
          begin
               gain := CalculateGain(Weights, dataSetIdx, i, splitVal, entropy, numClasses, numElements);

               if (gain > maxGain) or ((gain = maxGain) and (i <> fLastFeatureIdx)) then
               begin
                    maxSplitFeatureIdx := i;
                    maxSplitVal := splitVal;
                    maxGain := gain;
                    maxnumElements := numElements;
               end;
          end;

          // create the nodes
          SetLength(leftData, maxnumElements);
          SetLength(rightData, Length(dataSetIdx) - maxnumElements);
          leftIdx := 0;
          rightIdx := 0;

          // split data
          for i := 0 to Length(dataSetIdx) - 1 do
          begin
               if DataSet[dataSetIdx[i]].FeatureVec[maxSplitFeatureIdx] < maxSplitVal then
               begin
                    leftData[leftIdx] := dataSetIdx[i];
                    inc(leftIdx);
               end
               else
               begin
                    rightData[rightIdx] := dataSetIdx[i];
                    inc(rightIdx);
               end;
          end;

          // release memory, we don't want to use O(n^2) memory consumption (max O(2n)
          dataSetIdx := nil;

          // ##########################################################
          // #### Create result
          fLastFeatureIdx := maxSplitFeatureIdx;

          Result := TTreeNode.Create;
          TTreeNode(Result).LeftItem := SplitByProperty(weights, leftData, curDepth + 1);
          TTreeNode(Result).RightItem := SplitByProperty(weights, rightData, curDepth + 1);

          nodeData := T45NodeData.Create;
          nodeData.NumClasses := numClasses;
          nodeData.Classes := Copy(Classes, 1, NumClasses);
          nodeData.WeightedSums := Copy(WeightedNumElements, 1, NumClasses);
          nodeData.FeatureSplitIndex := maxSplitFeatureIdx;
          nodeData.SplitVal := maxSplitVal;
          nodeData.NumExamples := overallNumExamples;

          Result.TreeData := nodeData;

          // create array again
          SetLength(dataSetIdx, Length(leftData) + Length(rightData));
          if Length(leftData) > 0 then
             Move(leftData[0], dataSetIdx[0], sizeof(dataSetIdx[0])*Length(LeftData));
          if Length(rightData) > 0 then
             Move(rightData[0], dataSetIdx[Length(leftData)], sizeof(dataSetIdx[0])*Length(rightData));
     end;
end;

{ T45DecisionTree }

function TC45Classifier.Classify(Example: TCustomExample;
  var confidence: double): integer;
var node : TCustomTreeItem;
    nodeData : T45NodeData;
begin
     // todo: incorporate missing value detection
     node := fTree;

     while Assigned(node) and not (node is TTreeLeave) do
     begin
          nodeData := T45NodeData(node.TreeData);

          if Example.FeatureVec[nodeData.FeatureSplitIndex] < nodeData.SplitVal
          then
              node := TTreeNode(node).LeftItem
          else
              node := TTreeNode(node).RightItem;

          assert(Assigned(node), 'No leave assigned');
     end;

     assert(node is TTreeLeave, 'Error no leave found');
     nodeData := T45NodeData(node.TreeData);
     Result := nodeData.Classes[0];

     confidence := 0;
end;

constructor TC45Classifier.Create(Tree: TCustomTreeItem);
var dataObj : T45NodeData;
begin
     inherited Create;

     dataObj := T45NodeData(Tree.TreeData);
     fClasses := dataObj.Classes;
     fTree := Tree;
end;

procedure TC45Classifier.DefineProps;
var mem : TMemoryStream;
begin
     if Length(fClasses) > 0 then
        AddBinaryProperty(c45ClassesProps, fClasses[0], Length(fClasses)*sizeof(longInt));

     if Assigned(fTree) then
     begin
          mem := TMemoryStream.Create;
          try
             SaveTreeToStream(mem);

             AddBinaryProperty(c45TreeProps, mem.Memory^, mem.Size);
          finally
                 mem.Free;
          end;
     end;
end;

destructor TC45Classifier.Destroy;
begin
     fTree.Free;

     inherited;
end;

procedure TC45Classifier.LoadTreeFromStream(stream: TStream);
var version : integer;

function ReadFromStream(stream : TStream) : TCustomTreeItem;
var cTag : byte;
begin
     stream.ReadBuffer(cTag, sizeof(byte));
     if cTag = 0 then
     begin
          Result := TTreeNode.Create;
          Result.TreeData := T45NodeData.Create;
          T45NodeData(Result.TreeData).LoadFromStream(stream);
          // left right part
          TTreeNode(Result).LeftItem := ReadFromStream(Stream);
          TTreeNode(Result).RightItem := ReadFromStream(Stream);
     end
     else
     begin
          Result := TTreeLeave.Create;
          Result.TreeData := T45NodeData.Create;
          T45NodeData(Result.TreeData).LoadFromStream(stream);
     end;
end;

begin
     Stream.ReadBuffer(version, sizeof(version));
     assert(version = 0, 'No other version supported');

     fTree := ReadFromStream(Stream);
end;

procedure TC45Classifier.OnLoadBinaryProperty(const Name: String;
  const Value; size: integer);
var mem : TMemoryStream;
begin
     if CompareText(Name, c45ClassesProps) = 0 then
     begin
          if size > 0 then
          begin
               SetLength(fClasses, size div sizeof(LongInt));
               Move(Value, fClasses[0], Length(fClasses));
          end;
     end
     else if CompareText(Name, c45TreeProps) = 0 then
     begin
          Assert(not Assigned(fTree), 'error Tree already assigned');
          mem := TMemoryStream.Create;
          try
             mem.Write(Value, size);
             mem.Position := 0;
             LoadTreeFromStream(mem);
          finally
                 mem.Free;
          end;
     end;
end;

procedure TC45Classifier.SaveTreeToStream(stream: TStream);

procedure WriteNode(node : TCustomTreeItem);
var cTag : byte;
begin
     if node is TTreeNode then
     begin
          cTag := 0;
          stream.Write(ctag, sizeof(byte));
          T45NodeData(node.TreeData).WriteToStream(stream);

          WriteNode(TTreeNode(node).LeftItem);
          WriteNode(TTreeNode(node).RightItem);
     end
     else
     begin
          cTag := 1;
          stream.Write(ctag, sizeof(byte));

          T45NodeData(node.TreeData).WriteToStream(stream);
     end;
end;
var version : integer;
begin
     version := 0;
     Stream.WriteBuffer(version, sizeof(version));
     WriteNode(fTree);
end;

initialization
  RegisterMathIO(TC45Classifier);

end.
