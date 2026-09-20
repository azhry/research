# IFMS maintenance-request flow

## Responsibility and ordering

The maintenance-request UI (`fs-request`) starts the flow after the user confirms
the request. `ifms-middleware` creates the maintenance request, then calls the SAP
CPI IFMS flow to create the SAP notification. SAP information is created only
after that notification creation succeeds in SAP.

```mermaid
flowchart TD
    A[Maintenance Request UI / fs-request<br/>User confirms] --> B[ifms-middleware<br/>Create maintenance request]
    B --> C[ifms-middleware -> SAP CPI<br/>Create SAP notification through IFMS flow]
    C -->|Success| D[ifms-middleware<br/>Create SAP information]
    C -->|Failure| X[Stop SAP-information creation<br/>Report or handle notification failure]
    D --> E[SAP CPI<br/>Send maintenance-order event]
    E --> F[ifms-middleware HTTP endpoint<br/>Publish event]
    F --> G[.NET Channel<br/>Background worker consumes event]
    G --> H[ifms-middleware event handler<br/>Update maintenance request with MO number]
    H --> I[SAP information reflects the maintenance order]
```

## Invariant

SAP notification creation is the gate for SAP-information creation. If SAP CPI
does not successfully create the notification, the middleware must not create SAP
information and the maintenance-order event/update path must not be treated as
completed.
