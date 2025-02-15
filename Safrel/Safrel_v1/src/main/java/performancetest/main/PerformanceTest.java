package performancetest.main;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import performancetest.model.VirtualMachine;
import performancetest.service.ReinforcementLearning;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class PerformanceTest {

    public static void main(String[] args) throws InterruptedException {
        double VmsCap_CPU = 10.00; // %
        double VmsCap_Mem = 50.00; // GB
        double VmsCap_Disk = 1000.00; // GB
        double VmsCap_ResTime = 3000.00; // ms
        int NumofTestApps = 3;
        Double[] Requirement_ResTimes = new Double[NumofTestApps];
        Random rand1 = new Random();

        for (int i = 0; i < NumofTestApps; i++) {
            double ReqResTime = rand1.nextInt((int) (VmsCap_ResTime - 500) + 1) + 500;
            Requirement_ResTimes[i] = ReqResTime;
        }

        List VMList = new LinkedList();

        // CPU-Int Samples
        Double[] SenArray1 = { 0.70, 0.30, 0.00 };
        Double[] SenArray2 = { 0.90, 0.10, 0.00 };
        Double[] SenArray3 = { 0.80, 0.20, 0.00 };

        List SensitivityCollection_CPUTINT = new LinkedList();
        List SensitivityCollection_MemTINT = new LinkedList();
        List SensitivityCollection_DiskTINT = new LinkedList();

        SensitivityCollection_CPUTINT.add(SenArray1);
        SensitivityCollection_CPUTINT.add(SenArray2);
        SensitivityCollection_CPUTINT.add(SenArray3);

        InitializeVms(NumofTestApps, VmsCap_CPU, VmsCap_Mem, VmsCap_Disk, Requirement_ResTimes,
                SensitivityCollection_CPUTINT,
                VMList);

        System.out.println("VMs with various types of CPU Intensive applications have been initialized");

        // Learning agents have been built for testing different types of VMs including
        // CPU-Intensive, Mem-Intensive, Disk-Intensive (One agent per type) Agent1:
        // CPU-Int Agent2: Mem-Int Agent3: Disk-Int
        List LearningAgents = new LinkedList();

        {
            // Initializing The Qtable for CPU Int
            ReinforcementLearning RL1 = new ReinforcementLearning();
            RL1.InitializingstateActions();
            System.out.println("Agent1 " + "starts. It has been initialized");

            List LearningTrialpEpsilonList = new LinkedList();

            float epsilon = (float) 0.2;
            float Targetepsilon = (float) 0.2;
            float DecreaseStep = (epsilon - Targetepsilon) / (100 - 1);

            for (int i = 0; i < 1; i++) {
                float[] LearningTrailsperEpsilon = new float[2];
                float learningTrialsVar = 0;
                float EpsilonVal;

                System.out.println("epsilon= " + epsilon);

                VirtualMachine VM = new VirtualMachine();
                int index = i;
                double VM_CPU_i_val = ((VirtualMachine) VMList.get(0)).VM_CPU_i; // 3.0
                double VM_Mem_i_val = ((VirtualMachine) VMList.get(0)).VM_Mem_i; // 29
                double VM_Disk_i_val = ((VirtualMachine) VMList.get(0)).VM_Disk_i; // 175
                double Requirement_ResTime_val = ((VirtualMachine) VMList.get(0)).Requirement_ResTime; // 2692
                double ResponseTime_i_val = ((VirtualMachine) VMList.get(0)).ResponseTime_i; // 2070
                double Acceptolerance_val = ((VirtualMachine) VMList.get(0)).Acceptolerance; // 0.1
                double VM_CPU_g_val = ((VirtualMachine) VMList.get(0)).VM_CPU_g; // 3.0
                double VM_Mem_g_val = ((VirtualMachine) VMList.get(0)).VM_Mem_g; // 29.0
                double VM_Disk_g_val = ((VirtualMachine) VMList.get(0)).VM_Disk_g; // 175
                double VM_CPUtil_val = ((VirtualMachine) VMList.get(0)).VM_CPUtil; // 1.0
                double VM_MemUtil_val = ((VirtualMachine) VMList.get(0)).VM_MemUtil; // 1.0
                double VM_DiskUtil_val = ((VirtualMachine) VMList.get(0)).VM_DiskUtil; // 1.0

                VM.index = index;
                VM.VM_CPU_i = VM_CPU_i_val;
                VM.VM_Mem_i = VM_Mem_i_val;
                VM.VM_Disk_i = VM_Disk_i_val;
                VM.VM_SensitivityValues = ((VirtualMachine) VMList.get(0)).VM_SensitivityValues; // 0.97, 0.07, 0.0
                VM.Requirement_ResTime = Requirement_ResTime_val;
                VM.ResponseTime_i = ResponseTime_i_val;
                VM.Acceptolerance = Acceptolerance_val;
                VM.VM_CPU_g = VM_CPU_g_val;
                VM.VM_Mem_g = VM_Mem_g_val;
                VM.VM_Disk_g = VM_Disk_g_val;
                VM.ResponseTime = ResponseTime_i_val;
                VM.NormalizedResponsetime = 0.0;
                VM.Throughput = 0.0;
                VM.VM_CPUtil = VM_CPUtil_val;
                VM.VM_MemUtil = VM_MemUtil_val;
                VM.VM_DiskUtil = VM_DiskUtil_val;

                // detecting the Current State
                List DetectedState_C;
                DetectedState_C = RL1.DetectState(VM);

                // extracting the Index of state with Max Membership degree
                // finding the Index of Current State in the QTable
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
                VM.ResponseTime = ResponseTime_i_val;

                while ((1.2 * VM.Requirement_ResTime) > VM.ResponseTime) {
                    IndexofCurrentState = RL1.Learn(IndexofCurrentState, VM, epsilon);
                    learningTrialsVar++;

                }

                EpsilonVal = epsilon;
                LearningTrailsperEpsilon[1] = learningTrialsVar;
                LearningTrailsperEpsilon[0] = EpsilonVal;
                LearningTrialpEpsilonList.add(LearningTrailsperEpsilon);

                System.out.println("The Test agent for VM0 has converged:");
                System.out.println("Initial external Conditions ->" + " CPU: " + VM.VM_CPU_i + " Memory: "
                        + VM.VM_Mem_i + " Disk: " + VM.VM_Disk_i);
                System.out.println("Test case: CPU: " + Math.round(VM.VM_CPU_g * 100.0) / 100.0 + " Mem: "
                        + Math.round(VM.VM_Mem_g * 100.0) / 100.0 + " Disk: "
                        + Math.round(VM.VM_Disk_g * 100.0) / 100.0);
                System.out.println("**************************************************");
                java.lang.Thread.sleep(1000);
            }

            {
                epsilon = (float) 0.2;
                Targetepsilon = (float) 0.2;
                DecreaseStep = (epsilon - Targetepsilon) / (VMList.size() - 2);
                List LearningTrialpEpsilonList2 = new LinkedList();

                for (int i = 1; i < VMList.size(); i++) {
                    float[] LearningTrailsperEpsilon2 = new float[5];
                    float learningTrialsVar2 = 0;
                    float EpsilonVal2;
                    float Similarity1 = 0;
                    float Similarity2 = 0;
                    float Similarity3 = 0;

                    VirtualMachine VM2 = new VirtualMachine();
                    VM2.index = i;
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

                    // measuring Similarity

                    Double[] SenArrayA = VM2.VM_SensitivityValues;
                    Double[] SenArrayB = ((VirtualMachine) VMList.get(i - 1)).VM_SensitivityValues;

                    Double Similarity1_Part1 = (SenArrayA[0] * SenArrayB[0]) + (SenArrayA[1] * SenArrayB[1])
                            + (SenArrayA[2] * SenArrayB[2]);
                    Double Similarity1_Part2 = Math
                            .sqrt(Math.pow(SenArrayA[0], 2) + Math.pow(SenArrayA[1], 2) + Math.pow(SenArrayA[2], 2))
                            * Math.sqrt(
                                    Math.pow(SenArrayB[0], 2) + Math.pow(SenArrayB[1], 2) + Math.pow(SenArrayB[2], 2));
                    Similarity1 = (float) (Similarity1_Part1 / Similarity1_Part2);

                    if (i > 1) {
                        Double[] SenArrayC = ((VirtualMachine) VMList.get(i - 2)).VM_SensitivityValues;
                        Double Similarity2_Part1 = (SenArrayA[0] * SenArrayC[0]) + (SenArrayA[1] * SenArrayC[1])
                                + (SenArrayA[2] * SenArrayC[2]);
                        Double Similarity2_Part2 = Math
                                .sqrt(Math.pow(SenArrayA[0], 2) + Math.pow(SenArrayA[1], 2) + Math.pow(SenArrayA[2], 2))
                                * Math.sqrt(Math.pow(SenArrayC[0], 2) + Math.pow(SenArrayC[1], 2)
                                        + Math.pow(SenArrayC[2], 2));
                        Similarity2 = (float) (Similarity2_Part1 / Similarity2_Part2);

                        if (i > 2) {
                            Double[] SenArrayD = ((VirtualMachine) VMList.get(i - 3)).VM_SensitivityValues;
                            Double Similarity3_Part1 = (SenArrayA[0] * SenArrayD[0]) + (SenArrayA[1] * SenArrayD[1])
                                    + (SenArrayA[2] * SenArrayD[2]);
                            Double Similarity3_Part2 = Math.sqrt(
                                    Math.pow(SenArrayA[0], 2) + Math.pow(SenArrayA[1], 2) + Math.pow(SenArrayA[2], 2))
                                    * Math.sqrt(Math.pow(SenArrayD[0], 2) + Math.pow(SenArrayD[1], 2)
                                            + Math.pow(SenArrayD[2], 2));
                            Similarity3 = (float) (Similarity3_Part1 / Similarity3_Part2);
                        }
                    }

                    LearningTrailsperEpsilon2[2] = Similarity1;
                    LearningTrailsperEpsilon2[3] = Similarity2;
                    LearningTrailsperEpsilon2[4] = Similarity3;

                    System.out.println("epsilon= " + epsilon);

                    // detecting the Current State
                    List DetectedState_C;
                    DetectedState_C = RL1.DetectState(VM2);

                    // extracting the Index of state with Max Membership degree
                    // finding the Index of Current State in the QTable
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
                    while ((1.5 * VM2.Requirement_ResTime) > VM2.ResponseTime) {
                        IndexofCurrentState = RL1.Learn(IndexofCurrentState, VM2, epsilon);
                        learningTrialsVar2++;

                    }

                    EpsilonVal2 = epsilon;
                    LearningTrailsperEpsilon2[1] = learningTrialsVar2;
                    LearningTrailsperEpsilon2[0] = EpsilonVal2;

                    LearningTrialpEpsilonList2.add(LearningTrailsperEpsilon2);

                    System.out.println("Initial external Conditions VM" + i + " ->" + " CPU: " + VM2.VM_CPU_i
                            + " Memory: " + VM2.VM_Mem_i + " Disk: " + VM2.VM_Disk_i);
                    System.out.println("Test case: CPU: " + Math.round(VM2.VM_CPU_g * 100.0) / 100.0 + " Mem: "
                            + Math.round(VM2.VM_Mem_g * 100.0) / 100.0 + " Disk: "
                            + Math.round(VM2.VM_Disk_g * 100.0) / 100.0);
                    System.out.println("**************************************************");
                    java.lang.Thread.sleep(1000);

                }

            }
        }
    }

    public static void InitializeVms(int n, double VmsCap_CPU, double VmsCap_Mem, double VmsCap_Disk,
            Double[] Requirement_ResTimes, List SensitivityCollection, List VMList) {
        Random rand = new Random();

        for (int i = 0; i < n; i++) {
            int VM_CPU = (int) (Math.random() * VmsCap_CPU + 1);
            int VM_Mem = (int) (Math.random() * VmsCap_Mem + 1);
            int VM_Disk = rand.nextInt((int) (VmsCap_Disk - 100) + 1) + 100;
            int VM_SenIndex = (int) (Math.random() * SensitivityCollection.size());
            int VM_RequiredResTimeIndex = (int) (Math.random() * 3);
            double VM_ResTime = Math.floor((Math.random()
                    * (Requirement_ResTimes[VM_RequiredResTimeIndex]
                            - (Requirement_ResTimes[VM_RequiredResTimeIndex] / 2.0))
                    + (Requirement_ResTimes[VM_RequiredResTimeIndex] / 2.0)));
            System.out.println("Initial ResPonse Time: " + VM_ResTime);
            System.out.println("required Responsetime:" + Requirement_ResTimes[VM_RequiredResTimeIndex]);
            VirtualMachine VM1 = new VirtualMachine();
            VM1.VM_CPU_i = VM_CPU;
            VM1.VM_Mem_i = VM_Mem;
            VM1.VM_Disk_i = VM_Disk;
            VM1.VM_SensitivityValues = (Double[]) SensitivityCollection.get(VM_SenIndex);
            VM1.Requirement_ResTime = Requirement_ResTimes[VM_RequiredResTimeIndex];
            VM1.ResponseTime_i = VM_ResTime;
            VM1.Acceptolerance = 0.1;
            VM1.VM_CPU_g = VM1.VM_CPU_i;
            VM1.VM_Mem_g = VM1.VM_Mem_i;
            VM1.VM_Disk_g = VM1.VM_Disk_i;
            VM1.VM_CPUtil = 1.0;
            VM1.VM_MemUtil = 1.0;
            VM1.VM_DiskUtil = 1.0;
            VM1.ResponseTime = VM_ResTime;
            VM1.Throughput = 0.0;
            VM1.NormalizedResponsetime = 0.0;

            VMList.add(VM1);

        }
    }

    public static void WriteToExcel(List TrialsperEpsilon, int SheetNum) {
        XSSFWorkbook workbook = new XSSFWorkbook();
        XSSFSheet sheet;

        if (SheetNum == 1)
            sheet = workbook.createSheet("First Agent Learning Efficiency");
        else
            sheet = workbook.createSheet("Later Agnets Learning Efficiency");

        int rowNum = 0;

        for (int i = 0; i < TrialsperEpsilon.size(); i++) {
            Row row = sheet.createRow(++rowNum);

            int columnCount = 0;

            Cell cell = row.createCell(columnCount);
            cell.setCellValue(((float[]) TrialsperEpsilon.get(i))[0]);
            columnCount++;
            Cell cell_1 = row.createCell(columnCount);
            cell_1.setCellValue(((float[]) TrialsperEpsilon.get(i))[1]);
            if (SheetNum != 1) {
                columnCount++;
                Cell cell_2 = row.createCell(columnCount);
                cell_2.setCellValue(((float[]) TrialsperEpsilon.get(i))[2]);

                columnCount++;
                Cell cell_3 = row.createCell(columnCount);
                cell_3.setCellValue(((float[]) TrialsperEpsilon.get(i))[3]);
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
