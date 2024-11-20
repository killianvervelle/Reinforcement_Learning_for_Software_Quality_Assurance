package performancetest;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;


public class PerformanceTest {

    public static void main(String[] args) throws InterruptedException, IOException {
        int VmsCap_CPU = 50; //%
        int VmsCap_Mem = 16; //GB
        int VmsCap_Disk = 20; //GB
        double VmsCap_ResTime = 3000.00; //ms
        Double[] Requirement_ResTimes = {VmsCap_ResTime, VmsCap_ResTime, VmsCap_ResTime};
        Double[] Initial_ResTimes = {333.0, 205.0, 145.0};

        List VMList = new LinkedList();

        List SensitivityCollection = new LinkedList();

        //Sensitivity values {CPU, Memory, Disk util}
        Double[] SenArray1 = {0.97, 0.03, 0.00};
        Double[] SenArray2 = {0.95, 0.05, 0.00};
        Double[] SenArray3 = {0.3, 0.7, 0.0};

        SensitivityCollection.add(SenArray1);
        SensitivityCollection.add(SenArray2);
        SensitivityCollection.add(SenArray3);

        InitializeVms(SensitivityCollection.size(), VmsCap_CPU, VmsCap_Mem, VmsCap_Disk, Requirement_ResTimes, SensitivityCollection, Initial_ResTimes, VMList);

        System.out.println("VMs with various types of CPU Intensive applications have been initialized");

        List LearningAgents = new LinkedList();

        List LearningTrialpEpsilonList = new LinkedList();

        float epsilon = (float) 0.8;
        float Targetepsilon = (float) 0.2;
        float DecreaseStep = (epsilon - Targetepsilon) / (100 - 1);

        {
            int i = 0;

            float[] LearningTrailsperEpsilon = new float[2];
            float learningTrialsVar = 0;
            float EpsilonVal;

            System.out.println("Initial epsilon value= " + epsilon);

            VirtualMachine VM = (VirtualMachine) VMList.get(0);

            ReinforcementLearning RL1 = new ReinforcementLearning(0, VM);
            RL1.InitializingstateActions();
            System.out.println("Initialized RL agent 1.");

            //Detecting the Current State
            List DetectedState_C;
            DetectedState_C = RL1.DetectState(VM);

            // Extracting the Index of state with Max Membership degree
            // Finding the Index of Current State in the QTable
            int IndexofCurrentState = 0;

            List FinalDetectedState = new LinkedList();

            double MaxMemdegree = 0.0;
            String[] pair = new String[2];

            for (Object StateMember : DetectedState_C) {
                double Degree = Double.parseDouble(((String[]) StateMember)[1]);
                if (Degree > MaxMemdegree) {
                    pair[0] = ((String[]) StateMember)[0];
                    pair[1] = ((String[]) StateMember)[1];
                    MaxMemdegree = Degree;
                }
            }
            FinalDetectedState.add(pair);

            if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLLL"))
                IndexofCurrentState = 0;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLLA"))
                IndexofCurrentState = 1;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLLH"))
                IndexofCurrentState = 2;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLHL"))
                IndexofCurrentState = 3;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLHA"))
                IndexofCurrentState = 4;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLHH"))
                IndexofCurrentState = 5;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHLL"))
                IndexofCurrentState = 6;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHLA"))
                IndexofCurrentState = 7;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHLH"))
                IndexofCurrentState = 8;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHHL"))
                IndexofCurrentState = 9;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHHA"))
                IndexofCurrentState = 10;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHHH"))
                IndexofCurrentState = 11;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLLL"))
                IndexofCurrentState = 12;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLLA"))
                IndexofCurrentState = 13;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLLH"))
                IndexofCurrentState = 14;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLHL"))
                IndexofCurrentState = 15;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLHA"))
                IndexofCurrentState = 16;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLHH"))
                IndexofCurrentState = 17;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHLL"))
                IndexofCurrentState = 18;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHLA"))
                IndexofCurrentState = 19;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHLH"))
                IndexofCurrentState = 20;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHHL"))
                IndexofCurrentState = 21;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHHA"))
                IndexofCurrentState = 22;
            else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHHH"))
                IndexofCurrentState = 23;

            System.out.println("Current State: " + IndexofCurrentState);
            VM.ResponseTime = VM.ResponseTime_i;

            while (VM.Requirement_ResTime > VM.ResponseTime) {
                System.out.println("Required response time: " + VM.Requirement_ResTime + ", Actual Response time: " + VM.ResponseTime);
                IndexofCurrentState = RL1.Learn(IndexofCurrentState, VM, epsilon);
                learningTrialsVar++;
                i++;
            }

            EpsilonVal = epsilon;
            LearningTrailsperEpsilon[1] = learningTrialsVar;
            LearningTrailsperEpsilon[0] = EpsilonVal;
            LearningTrialpEpsilonList.add(LearningTrailsperEpsilon);

            if (i == 99) {
                System.out.println("The Test agent for VM0 has converged:");
                System.out.println("Initial external Conditions ->" + " CPU: " + VM.VM_CPU_i + " Memory: " + VM.VM_Mem_i + " Disk: " + VM.VM_Disk_i);
                System.out.println("Test case: CPU: " + VM.VM_CPU_g + " Mem: " + VM.VM_Mem_g + " Disk: " + VM.VM_Disk_g);
                System.out.println("**************************************************");
                java.lang.Thread.sleep(1000);
            }
        }

        //WriteToExcel (LearningTrialpEpsilonList,1);

        {
            epsilon = (float) 0.8;
            Targetepsilon = (float) 0.2;
            DecreaseStep = (epsilon - Targetepsilon) / (VMList.size() - 2);
            List LearningTrialpEpsilonList2 = new LinkedList();

            System.out.println("----------------------------------------");
            System.out.println("NEW SUT LOADING...");
            System.out.println("----------------------------------------");


            for (int i = 1; i < VMList.size(); i++) {
                // Adaptive Epsilon
                // if (i==1)
                //   epsilon =(float) 0.85;
                // else
                //   if (epsilon > 0.2)
                //      epsilon = ((float) Math.round((epsilon-DecreaseStep)*1000.0)/1000);}

                float[] LearningTrailsperEpsilon2 = new float[5];
                float learningTrialsVar2 = 0;
                float EpsilonVal2;
                float Similarity1 = 0;
                float Similarity2 = 0;
                float Similarity3 = 0;
                int j = 0;

                VirtualMachine VM2 = new VirtualMachine();
                VM2.VM_CPU_i = ((VirtualMachine) VMList.get(i)).VM_CPU_i;
                VM2.VM_Mem_i = ((VirtualMachine) VMList.get(i)).VM_Mem_i;
                VM2.VM_Disk_i = ((VirtualMachine) VMList.get(i)).VM_Disk_i;
                VM2.VM_SensitivityValues = ((VirtualMachine) VMList.get(i)).VM_SensitivityValues;
                VM2.Requirement_ResTime = ((VirtualMachine) VMList.get(i)).Requirement_ResTime;
                VM2.ResponseTime_i = ((VirtualMachine) VMList.get(i)).ResponseTime_i;
                VM2.Acceptolerance = ((VirtualMachine) VMList.get(i)).Acceptolerance;
                VM2.VM_CPU_g = ((VirtualMachine) VMList.get(i)).VM_CPU_g;
                VM2.VM_Mem_g = ((VirtualMachine) VMList.get(i)).VM_Mem_g;
                VM2.VM_Disk_g = ((VirtualMachine) VMList.get(i)).VM_Disk_g;
                VM2.ResponseTime = ((VirtualMachine) VMList.get(i)).ResponseTime_i;
                VM2.NormalizedResponsetime = 0.0;
                VM2.Throughput = 0.0;
                VM2.VM_CPUtil = ((VirtualMachine) VMList.get(i)).VM_CPUtil;
                VM2.VM_MemUtil = ((VirtualMachine) VMList.get(i)).VM_MemUtil;
                VM2.VM_DiskUtil = ((VirtualMachine) VMList.get(i)).VM_DiskUtil;

                //measuring similarity between VMs
                Double[] SenArrayA = VM2.VM_SensitivityValues;
                Double[] SenArrayB = ((VirtualMachine) VMList.get(i - 1)).VM_SensitivityValues;

                Double Similarity1_Part1 = (SenArrayA[0] * SenArrayB[0]) + (SenArrayA[1] * SenArrayB[1]) + (SenArrayA[2] * SenArrayB[2]);
                Double Similarity1_Part2 = Math.sqrt(Math.pow(SenArrayA[0], 2) + Math.pow(SenArrayA[1], 2) + Math.pow(SenArrayA[2], 2)) * Math.sqrt(Math.pow(SenArrayB[0], 2) + Math.pow(SenArrayB[1], 2) + Math.pow(SenArrayB[2], 2));
                Similarity1 = (float) (Similarity1_Part1 / Similarity1_Part2);

                if (i > 1) {
                    Double[] SenArrayC = ((VirtualMachine) VMList.get(i - 2)).VM_SensitivityValues;
                    Double Similarity2_Part1 = (SenArrayA[0] * SenArrayC[0]) + (SenArrayA[1] * SenArrayC[1]) + (SenArrayA[2] * SenArrayC[2]);
                    Double Similarity2_Part2 = Math.sqrt(Math.pow(SenArrayA[0], 2) + Math.pow(SenArrayA[1], 2) + Math.pow(SenArrayA[2], 2)) * Math.sqrt(Math.pow(SenArrayC[0], 2) + Math.pow(SenArrayC[1], 2) + Math.pow(SenArrayC[2], 2));
                    Similarity2 = (float) (Similarity2_Part1 / Similarity2_Part2);

                    if (i > 2) {
                        Double[] SenArrayD = ((VirtualMachine) VMList.get(i - 3)).VM_SensitivityValues;
                        Double Similarity3_Part1 = (SenArrayA[0] * SenArrayD[0]) + (SenArrayA[1] * SenArrayD[1]) + (SenArrayA[2] * SenArrayD[2]);
                        Double Similarity3_Part2 = Math.sqrt(Math.pow(SenArrayA[0], 2) + Math.pow(SenArrayA[1], 2) + Math.pow(SenArrayA[2], 2)) * Math.sqrt(Math.pow(SenArrayD[0], 2) + Math.pow(SenArrayD[1], 2) + Math.pow(SenArrayD[2], 2));
                        Similarity3 = (float) (Similarity3_Part1 / Similarity3_Part2);
                    }
                }

                LearningTrailsperEpsilon2[2] = Similarity1;
                LearningTrailsperEpsilon2[3] = Similarity2;
                LearningTrailsperEpsilon2[4] = Similarity3;

                System.out.println("epsilon= " + epsilon);

                List DetectedState_C;

                //Detecting the Current State
                ReinforcementLearning RL2 = new ReinforcementLearning(i, VM2);
                RL2.InitializingstateActions();
                System.out.println("Initialized RL agent " + i);

                DetectedState_C = RL2.DetectState(VM2);

                //extracting the Index of state with Max Membership degree
                //finding the Index of Current State in the QTable
                int IndexofCurrentState = 0;

                List FinalDetectedState = new LinkedList();

                Double MaxMemdegree = 0.0;
                String[] pair = new String[2];

                for (Object StateMember : DetectedState_C) {
                    Double Degree = Double.valueOf(((String[]) StateMember)[1]);
                    if (Degree > MaxMemdegree) {
                        pair[0] = ((String[]) StateMember)[0];
                        pair[1] = ((String[]) StateMember)[1];
                        MaxMemdegree = Degree;
                    }
                }
                FinalDetectedState.add(pair);

                if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLLL"))
                    IndexofCurrentState = 0;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLLA"))
                    IndexofCurrentState = 1;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLLH"))
                    IndexofCurrentState = 2;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLHL"))
                    IndexofCurrentState = 3;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLHA"))
                    IndexofCurrentState = 4;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LLHH"))
                    IndexofCurrentState = 5;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHLL"))
                    IndexofCurrentState = 6;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHLA"))
                    IndexofCurrentState = 7;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHLH"))
                    IndexofCurrentState = 8;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHHL"))
                    IndexofCurrentState = 9;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHHA"))
                    IndexofCurrentState = 10;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("LHHH"))
                    IndexofCurrentState = 11;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLLL"))
                    IndexofCurrentState = 12;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLLA"))
                    IndexofCurrentState = 13;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLLH"))
                    IndexofCurrentState = 14;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLHL"))
                    IndexofCurrentState = 15;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLHA"))
                    IndexofCurrentState = 16;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HLHH"))
                    IndexofCurrentState = 17;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHLL"))
                    IndexofCurrentState = 18;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHLA"))
                    IndexofCurrentState = 19;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHLH"))
                    IndexofCurrentState = 20;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHHL"))
                    IndexofCurrentState = 21;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHHA"))
                    IndexofCurrentState = 22;
                else if (((String[]) (FinalDetectedState.get(0)))[0].equals("HHHH"))
                    IndexofCurrentState = 23;

                System.out.println("Current State: " + IndexofCurrentState);

                VM2.ResponseTime = ((VirtualMachine) VMList.get(i)).ResponseTime_i;

                while (VM2.Requirement_ResTime > VM2.ResponseTime) {
                    IndexofCurrentState = RL2.Learn(IndexofCurrentState, VM2, epsilon);
                    learningTrialsVar2++;
                    j++;
                }

                EpsilonVal2 = epsilon;
                LearningTrailsperEpsilon2[1] = learningTrialsVar2;
                LearningTrailsperEpsilon2[0] = EpsilonVal2;

                if (j == 99) {
                    System.out.println("The Test agent for VM2 has converged:");
                    System.out.println("Initial external Conditions ->" + " CPU: " + VM2.VM_CPU_i + " Memory: " + VM2.VM_Mem_i + " Disk: " + VM2.VM_Disk_i);
                    System.out.println("Test case: CPU: " + VM2.VM_CPU_g + " Mem: " + VM2.VM_Mem_g + " Disk: " + VM2.VM_Disk_g);
                    System.out.println("**************************************************");
                    java.lang.Thread.sleep(1000);
                }

                LearningTrialpEpsilonList2.add(LearningTrailsperEpsilon2);

                System.out.println("Initial external Conditions VM" + i + " ->" + " CPU: " + VM2.VM_CPU_i + " Memory: " + VM2.VM_Mem_i + " Disk: " + VM2.VM_Disk_i);
                System.out.println("Test case: CPU: " + VM2.VM_CPU_g + " Mem: " + VM2.VM_Mem_g + " Disk: " + VM2.VM_Disk_g);
                System.out.println("**************************************************");
                java.lang.Thread.sleep(1000);

            }

            //WriteToExcel(LearningTrialpEpsilonList2, 2);

        }
    }

    public static void InitializeVms(int n, int VmsCap_CPU, long VmsCap_Mem, long VmsCap_Disk, Double[] Requirement_ResTimes, List SensitivityCollection, Double[] Initial_ResTimes, List VMList) {
        for (int i = 0; i < n; i++) {
            VirtualMachine VM1 = new VirtualMachine();
            VM1.VM_CPU_i = VmsCap_CPU;
            VM1.VM_Mem_i = VmsCap_Mem;
            VM1.VM_Disk_i = VmsCap_Disk;
            VM1.VM_SensitivityValues = (Double[]) SensitivityCollection.get(i);
            VM1.Requirement_ResTime = Requirement_ResTimes[i];
            VM1.ResponseTime_i = Initial_ResTimes[i];
            VM1.Acceptolerance = 0.1;
            VM1.VM_CPU_g = VM1.VM_CPU_i;
            VM1.VM_Mem_g = VM1.VM_Mem_i;
            VM1.VM_Disk_g = VM1.VM_Disk_i;
            VM1.VM_CPUtil = 1.0;
            VM1.VM_MemUtil = 1.0;
            VM1.VM_DiskUtil = 1.0;
            VM1.ResponseTime = 0.0;
            VM1.Throughput = 0.0;
            VM1.NormalizedResponsetime = 0.0;

            System.out.println("VM set up with initial CPU util of: " + VM1.VM_CPU_i
                    + ", Mem util of: " + VM1.VM_Mem_i
                    + ", Disk util of: " + VM1.VM_Disk_i
                    + " and Required Response Time of: " + Requirement_ResTimes[i]);

            VMList.add(VM1);

        }
    }

    public static void WriteToExcel(List TrialsperEpsilon, int SheetNum) {
        XSSFWorkbook workbook = new XSSFWorkbook();
        XSSFSheet sheet;

        if (SheetNum == 1)
            sheet = workbook.createSheet("First Agent Learning Efficiency");
        else
            sheet = workbook.createSheet("Later Agents Learning Efficiency");

        int rowNum = 0;

        for (Object o : TrialsperEpsilon) {
            Row row = sheet.createRow(++rowNum);

            int columnCount = 0;

            Cell cell = row.createCell(columnCount);
            cell.setCellValue(((float[]) o)[0]);
            columnCount++;
            Cell cell_1 = row.createCell(columnCount);
            cell_1.setCellValue(((float[]) o)[1]);
            if (SheetNum != 1) {
                columnCount++;
                Cell cell_2 = row.createCell(columnCount);
                cell_2.setCellValue(((float[]) o)[2]);
                columnCount++;
                Cell cell_3 = row.createCell(columnCount);
                cell_3.setCellValue(((float[]) o)[3]);
            }
        }

        try {
            FileOutputStream outputStream;
            if (SheetNum == 1)
                outputStream = new FileOutputStream("Gamma 0.9-1st Agent Learning efficiency-0.2 Homogeneous.xlsx");
            else
                outputStream = new FileOutputStream("Gamma 0.9-Later Agents Learning efficiency-0.2,Homogeneous.xlsx");

            workbook.write(outputStream);
            outputStream.close();
            System.out.println("wrote in file");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

