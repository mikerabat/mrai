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

unit IncrementalFisherLDA;

// #####################################################
// #### Incremental LDA classification
// #####################################################

interface

uses SysUtils, Classes, Types, BaseClassifier, BaseIncrementalLearner,
     FisherIncrementalClassifiers, FisherBatchLDA, Matrix, IncrementalPCA;

// #####################################################
// #### Incremental LDA algorithm
// Phd: Incremental, Robust, and Efficient Linear Discriminant Analysis Learning, page 39
// Augmented subspace learning without any vector reduction
type
  TFisherIncrementalLDA = class(TCustomIncrementalWeightedLearner)
  private
    fProps : TFisherAugmentedBaseProps;
    fClassifier : TIncrementalFisherLDAClassifier;
    function ExtractMatrixData(Examples : TCustomLearnerExampleList; var ownsMatrix: boolean): TDoubleMatrix;
    procedure ReduceAndAugmentBase(ldaClassifier : TIncrementalFisherLDAClassifier);
    procedure LDAOnA(ldaClassifier : TIncrementalFisherLDAClassifier; numClasses : integer; const classIdx : TIntIntArray; const classLabels : TIntegerDynArray);
  protected
    function GetClassifier : TCustomClassifier; override;
  public
    procedure OnLoadExample(Sender : TObject; Example : TCustomLearnerExample; const weight : double = -1); override;
    procedure OnLoadClass(Sender : TObject; Examples : TCustomLearnerExampleList; const Weights : TDoubleDynArray); override;
    procedure OnLoadInitComplete(Sender : TObject; Examples : TCustomLearnerExampleList; const Weights : TDoubleDynArray); override;

    procedure SetProperties(const props : TFisherAugmentedBaseProps);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
  end;

implementation

uses FisherClassifiers, BaseMatrixExamples, math, LinearAlgebraicEquations, MatrixConst;

// used to shortcut the copying process
type
  TFisherLDAClassifierCrack = class(TFisherLDAClassifier);

{ TFisherIncrementalLDA }

function TFisherIncrementalLDA.GetClassifier: TCustomClassifier;
procedure SetupLDAClassifier;
begin
     fClassifier := TIncrementalFisherLDAClassifier.Create;
     TIncrementalPCA(fClassifier.PCAData).NumEigenvectorsToKeep := fProps.NumLDAVectorsToKeep + 2;
end;
procedure SetupRobustLDAClassifier;
begin
     fClassifier := TFastRobustIncrementalLDAClassifier.Create;
     TFastRobustIncrementalPCA(fClassifier.PCAData).SetProperties(fProps.RobustPCAProps);
     TFastRobustIncrementalPCA(fClassifier.PCAData).NumEigenvectorsToKeep := fProps.NumLDAVectorsToKeep + 2;
end;
begin
     if not Assigned(fClassifier) then
     begin
          case fProps.ClassifierType of
            ctFast: SetupLDAClassifier;
            ctRobust: SetupRobustLDAClassifier;
            ctFastRobust: SetupRobustLDAClassifier;
          end;
     end;

     Result := fClassifier;
end;

procedure TFisherIncrementalLDA.LDAOnA(ldaClassifier : TIncrementalFisherLDAClassifier; numClasses : integer;
    const classIdx : TIntIntArray; const classLabels : TIntegerDynArray);
var x, y : integer;
    mu : TDoubleMatrix;
    Sb, Sw : TDoubleMatrix;
    tmp, tmpT : TDoubleMatrix;
    sumMtx : TDoubleMatrix;
    regularizeFact : double;
    U, S, V : TDoubleMatrix;
    classCenters : TDoubleMatrixDynArr;
    A : TDoubleMatrix;
begin
     mu := nil;
     sb := nil;
     sw := nil;
     U := nil;
     S := nil;
     V := nil;

     try
        // ##############################################################
        // #### Calculate Scatter matrices in the feature space
        if ldaClassifier is TFastRobustIncrementalLDAClassifier
        then
            A := TFastRobustIncrementalPCA(ldaClassifier.PCAData).A
        else
            A := TIncrementalPCA(ldaClassifier.PCAData).A;

        mu := A.Mean(True);
        SetLength(Classcenters, numClasses);

        for y := 0 to numClasses - 1 do
        begin
             classCenters[y] := TDoubleMatrix.Create(1, mu.Height);

             for x := 1 to classIdx[y][0] do
             begin
                  A.SetSubMatrix(classIdx[y][x], 0, 1, A.Height);
                  classCenters[y].AddInplace(A);
             end;

             classCenters[y].ScaleInPlace(1/classIdx[y][0]);
        end;

        // between class scatter matrix on A:
        Sb := TDoubleMatrix.Create(classCenters[0].Height, classCenters[0].Height);

        for y := 0 to numClasses - 1 do
        begin
             tmp := classCenters[y].Sub(mu);
             try
                tmpT := tmp.Transpose;
                try
                   tmp.MultInPlace(tmpT);
                   tmp.ScaleInPlace(classIdx[y][0]);
                   sb.AddInplace(tmp);
                finally
                       tmpT.Free;
                end;
             finally
                    tmp.Free;
             end;
        end;

        FreeAndNil(mu);

        // within class scatter matrix on A
        Sw := TDoubleMatrix.Create(classCenters[0].Height, classCenters[0].Height);
        for y := 0 to numClasses - 1 do
        begin
             sumMtx := TDoubleMatrix.Create(classCenters[0].Height, classCenters[0].Height);
             try
                for x := 1 to classIdx[y][0] do
                begin
                     A.SetSubMatrix(classIdx[y][x], 0, 1, A.Height);
                     tmp := A.Sub(classCenters[y]);
                     try
                        //tmp.ScaleInPlace(sqrt(Weights[classIdx[y][x]]));
                        tmp.ScaleInPlace(sqrt(1/classIdx[y][0]));
                        tmpT := tmp.Transpose;
                        try
                           tmp.MultInPlace(tmpT);
                        finally
                               tmpT.Free;
                        end;
                        sumMtx.AddInplace(tmp);
                     finally
                            tmp.Free;
                     end;
                end;

                Sw.AddInplace(sumMtx);
             finally
                    sumMtx.Free;
             end;
        end;

        // #############################################################
        // #### Perform LDA on A and on the calculated scatter matrices
        // regularize within class scatter matrix which ensures that the matrix can be inverted
        regularizeFact := Sw[0, 0];
        for y := 1 to Sw.Width - 1 do
            regularizeFact := Max(regularizeFact, Sw[y, y]);
        regularizeFact := Max(MinDouble*10000, regularizeFact*0.00001);
        for y := 0 to Sw.Width - 1 do
            Sw[y, y] := Sw[y, y] + regularizeFact;

        // now we can invert the  matrix without problems and calculate the LDA criterium
        if Sw.InvertInPlace <> leOk then
           raise ELDAException.Create('Error singular Sw');
        Sw.MultInPlace(Sb);
        FreeAndNil(Sb);
        if Sw.SVD(U, V, S) <> srOk then
           raise ELDAException.Create('Error LDA matrix cannot be computed');

        FreeAndNil(V);
        FreeAndNil(S);
        FreeAndNil(Sw);

        // the (numClasses - 1) eigenvectors of U matrix contains our LDA vectors
        U.SetSubMatrix(0, 0, numClasses - 1, U.Height);
        ldaclassifier.V.Assign(U, True);
        FreeAndNil(U);

        // project pca class centers to lda space to obtain vi's
        ldaclassifier.V.TransposeInPlace;
        for y := 0 to numClasses - 1 do
        begin
             tmp := ldaclassifier.V.Mult(classCenters[y]);
             classCenters[y].Free;
             classCenters[y] := tmp;
        end;

        // remove class centers from the previous steps
        for x := 0 to Length(ldaClassifier.Classcenters) - 1 do
            ldaClassifier.Classcenters[x].Free;

        ldaClassifier.ClassCenters := classCenters;

        A.UseFullMatrix;

        // ##############################################################
        // #### Reduce and augment pca space -> disciriminative information is kept!
        if ldaClassifier.PCAData.EigVecs.Width > fProps.NumLDAVectorsToKeep then
           ReduceAndAugmentBase(ldaClassifier);
     except
           for y := 0 to numClasses - 1 do
               FreeAndNil(classCenters[y]);

           FreeAndNil(mu);
           FreeAndNil(Sb);
           FreeAndNil(Sw);
           FreeAndNil(U);
           FreeAndNil(S);
           FreeAndNil(V);
     end;
end;

procedure TFisherIncrementalLDA.ReduceAndAugmentBase(ldaClassifier : TIncrementalFisherLDAClassifier);
var vhat : TDoubleMatrix;
    U, V, W : TDoubleMatrix;
    M : TDoubleMatrix;
    x : integer;
    y : integer;
    A : TDoubleMatrix;
begin
     ldaClassifier.V.TransposeInPlace;

     vhat := TDoubleMatrix.Create;
     try
        ldaClassifier.V.SetSubMatrix(0, fProps.NumLDAVectorsToKeep, ldaClassifier.V.Width, ldaClassifier.V.Height - fProps.NumLDAVectorsToKeep);
        vhat.Assign(ldaClassifier.V, True);
        ldaClassifier.V.UseFullMatrix;

        // orthogonalize splitted base -> vhatj. The base formula is vhat = v*(v'*v)^-0.5
        if vhat.SVD(U, V, W) <> srOk then
           raise ELDAException.Create('Error orthogonalizing matrix');

        try
           W.Free;

           // v is returned as v transpose
           //V.TransposeInPlace;
           FreeAndNil(vhat);

           // orthogonalize
           vhat := U.Mult(V);
        finally
               U.Free;
               V.Free;
        end;

        // #######################################################
        // #### Transformation matrix: M [ Ik       0k  ]
        //                               [ 0(n-k)   Vhat]
        M := TDoubleMatrix.Create(fProps.NumLDAVectorsToKeep + vhat.Width, fProps.NumLDAVectorsToKeep + vhat.Height);

        for x := 0 to fProps.NumLDAVectorsToKeep - 1 do
            M[x, x] := 1;

        for y := fProps.NumLDAVectorsToKeep to M.Height - 1 do
            for x := fProps.NumLDAVectorsToKeep to M.Width - 1 do
                M[x, y] := vhat[x - fProps.NumLDAVectorsToKeep, y - fProps.NumLDAVectorsToKeep];

        FreeAndNil(vhat);

        // #######################################################
        // #### Augmanted basis for pcaU is already transposed
        ldaClassifier.pcaData.EigVecs.MultInPlace(M);

        // #######################################################
        // #### Augmanted basis for pcaU:  Uhat = [Uk Un-k*Vhat]
        M.TransposeInPlace;
        M.MultInPlace(ldaClassifier.V);
        ldaClassifier.V.Assign(M, True);
        FreeAndNil(M);

        // #######################################################
        // #### estimate expected error distribution.
        // details in: Bischof, Leonardis: Robust Recognition using Eigenimages
        ldaClassifier.theta := 0;
        for y := fProps.NumLDAVectorsToKeep to ldaClassifier.pcaData.EigVals.Height - 1 do
            ldaClassifier.theta := ldaClassifier.theta + ldaClassifier.pcaData.EigVals[0, y];
        ldaClassifier.theta := ldaClassifier.theta*2/(ldaClassifier.pcaData.EigVals.Height);

        // backtranspose
        ldaClassifier.V.TransposeInPlace;

        // #######################################################
        // #### Reduce space of A (if necessary)
        if ldaClassifier is TFastRobustIncrementalLDAClassifier
        then
            A := TFastRobustIncrementalPCA(ldaClassifier.PCAData).A
        else
            A := TIncrementalPCA(ldaClassifier.PCAData).A;
        if ldaClassifier.PCAData.EigVecs.Width < A.Height then
           A.SetSubMatrix(0, 0, A.Width, ldaClassifier.PCAData.EigVecs.Width);
     except
           FreeAndNil(vhat);
           FreeAndNil(M);

           raise;
     end;
end;

procedure TFisherIncrementalLDA.OnLoadClass(Sender: TObject;
  Examples: TCustomLearnerExampleList; const Weights: TDoubleDynArray);
begin
     raise EAbstractError.Create('Not yet implemented');
end;

procedure TFisherIncrementalLDA.OnLoadExample(Sender: TObject;
  Example: TCustomLearnerExample; const weight: double);
var ldaClassifier : TIncrementalFisherLDAClassifier;
    mtx : TDoubleMatrix;
    i : Integer;
    numClasses : integer;
    classIdx : TIntIntArray;
    classLabels : TIntegerDynArray;
begin
     // #########################################################
     // #### First update the remaining PCA space
     ldaClassifier := fClassifier;
     mtx := TDoubleMatrix.Create(1, Example.FeatureVec.FeatureVecLen);
     try
        for i := 0 to mtx.Height - 1 do
            mtx[0, i] := Example.FeatureVec[i];

        if weight < 0 then
        begin
             if ldaClassifier is TFastRobustIncrementalLDAClassifier
             then
                 TFastRobustIncrementalPCA(ldaClassifier.PCAData).UpdateEigenspace(mtx)
             else
                 TIncrementalPCA(ldaClassifier.PCAData).UpdateEigenspace(mtx);
        end
        else
        begin
             if ldaClassifier is TFastRobustIncrementalLDAClassifier
             then
                 TFastRobustIncrementalPCA(ldaClassifier.PCAData).UpdateEigenspaceWeighted(mtx, weight)
             else
                 TIncrementalPCA(ldaClassifier.PCAData).UpdateEigenspaceWeighted(mtx, weight);
        end;
     finally
            mtx.Free;
     end;

     // #########################################################
     // #### Perform LDA on A
     fClassifier.AddClassIdx(Example.ClassVal);

     numClasses := IndexOfClasses(fClassifier.ClassIdx, fClassifier.NumLabels, classIdx, classLabels);
     ldaClassifier.ClassLabels := classLabels;

     LDAOnA(fClassifier, numClasses, classIdx, classLabels);
end;

class function TFisherIncrementalLDA.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := (Classifier = TIncrementalFisherLDAClassifier) or (Classifier = TFastRobustIncrementalLDAClassifier);
end;

function TFisherIncrementalLDA.ExtractMatrixData(Examples : TCustomLearnerExampleList; var ownsMatrix: boolean): TDoubleMatrix;
var x, y : integer;
begin
     if Examples is TMatrixLearnerExampleList
     then
         Result := TMatrixLearnerExampleList(Examples).Matrix
     else
     begin
          // Create a matrix of the feature vectors -> make it compatible with the pca routines
          Result := TDoubleMatrix.Create(Examples.Count, Examples[0].FeatureVec.FeatureVecLen);
          for y := 0 to Examples[0].FeatureVec.FeatureVecLen - 1 do
              for x := 0 to DataSet.Count - 1 do
                  Result[x, y] := Examples[x].FeatureVec[y];
     end;

     ownsMatrix := not (Examples is TMatrixLearnerExampleList);
end;


procedure TFisherIncrementalLDA.OnLoadInitComplete(Sender: TObject;
  Examples: TCustomLearnerExampleList; const Weights: TDoubleDynArray);
var matrixData : TDoubleMatrix;
    y : integer;
    mu : TDoubleMatrix;
    Sb, Sw : TDoubleMatrix;
    U, S, V : TDoubleMatrix;
    ownsMatrix : boolean;
    ldaClassifier : TIncrementalFisherLDAClassifier;
    classCenters : TDoubleMatrixDynArr;
    numClasses : integer;
    classIdx : TIntIntArray;
    classLabels : TIntegerDynArray;
    classLabelIdx : TIntegerDynArray;
    i : Integer;
begin
     // todo: introduce example weighting
     matrixData := nil;

     ldaClassifier := Classifier as TIncrementalFisherLDAClassifier;

     SetLength(classLabelIdx, Examples.Count);
     for i := 0 to Examples.Count - 1 do
         classLabelIdx[i] := Examples[i].ClassVal;
     ldaClassifier.ClassIdx := classLabelIdx;

     numClasses := IndexOfClasses(Examples, classIdx, classLabels);
     ldaClassifier.ClassLabels := classLabels;

     // ###################################################################
     // #### get indexes of the classes and the number of classes
     assert(numClasses > 0, 'Error no data');

     // ###################################################################
     // #### shrink the example space such (Fisher LDA)
     try
        matrixData := ExtractMatrixData(Examples, ownsMatrix);
        try
           if Length(weights) = 0 then
           begin
                if not ldaClassifier.PCAData.PCA(matrixData, 1, True) then
                   raise ELDAException.Create('Error: unable to calculate initial pca on data');
           end
           else
           begin
                if not ldaClassifier.PCAData.TemporalWeightPCA(matrixData, 1, True, Weights) then
                   raise ELDAException.Create('Error: unable to calculate initial pca on data');
           end;
        finally
               if ownsMatrix then
                  FreeAndNil(matrixData);
        end;

        // ##############################################################
        // #### Calculate LDA on A
        LDAOnA(ldaClassifier, numClasses, classIdx, classLabels);
     except
           for y := 0 to numClasses - 1 do
               FreeAndNil(classCenters[y]);

           FreeAndNil(mu);
           FreeAndNil(Sb);
           FreeAndNil(Sw);
           FreeAndNil(U);
           FreeAndNil(S);
           FreeAndNil(V);
     end;
end;

procedure TFisherIncrementalLDA.SetProperties(
  const props: TFisherAugmentedBaseProps);
begin
     fProps := props;
end;

end.
