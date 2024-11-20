package reload.model;


public class Transaction {

    public String name;
    public int workLoad;

    public Transaction(String name, int workLoad) {
        this.name = name;
        this.workLoad = workLoad;
    }

    @Override
    public String toString() {
        return this.name + ": " + workLoad;
    }
}
